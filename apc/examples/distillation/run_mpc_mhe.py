import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

from pyomo.dae import ContinuousSet
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_time_varying_target,
)
from pyomo.contrib.mpc.examples.cstr.model import (
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)
from apc.examples.distillation.run_mpc import get_steady_state_data
from apc.examples.distillation.model import (
    create_instance,
    create_model_bounds,
)
from apc.mhe_utils import (
    get_parameters_from_variables,
    get_disturbed_constraint_and_residual_expression,
    slice_components,
    get_tracking_cost_from_target_trajectory,
)


def run_cstr_mpc_mhe(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=20,
    samples_per_estimator_horizon=10,
    sample_time=60.0,
    ntfe_per_sample_controller=1,
    ntfe_per_sample_estimator=1,
    ntfe_plant=1,
    simulation_steps=5,
    tee=False,
):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    # create_model_bounds(m_controller) # optional in this example
    controller_interface = mpc.DynamicModelInterface(
        m_controller, m_controller.time
    )
    t0_controller = m_controller.time.first()

    estimator_horizon = sample_time * samples_per_estimator_horizon
    ntfe = ntfe_per_sample_estimator * samples_per_estimator_horizon
    m_estimator = create_instance(horizon=estimator_horizon, ntfe=ntfe)
    # create_model_bounds(m_estimator) # optional in this example
    estimator_interface = mpc.DynamicModelInterface(
        m_estimator, m_estimator.time
    )
    t0_estimator = m_estimator.time.first()

    m_plant = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)

    # Sets initial conditions and initializes
    controller_interface.load_data(initial_data)
    estimator_interface.load_data(initial_data)
    plant_interface.load_data(initial_data)

    #
    # Add objective to controller model
    #
    setpoint_variables = [m_controller.M[:, j] for j in m_controller.tray] + \
                            [m_controller.x[:, j] for j in m_controller.tray] + \
                                [m_controller.Qr, m_controller.Rec]
    weight_data = mpc.ScalarData(
        {**{m_controller.M[:, j]: 1.0 for j in m_controller.tray},
         **{m_controller.x[:, j]: 1.0E6 for j in m_controller.tray},
         m_controller.Qr: 1.0E-6,
         m_controller.Rec: 1.0E5,}
    )
    vset, tr_cost = controller_interface.get_penalty_from_target(
        setpoint_data, variables=setpoint_variables, weight_data=weight_data
    )
    m_controller.setpoint_set = vset
    m_controller.tracking_cost = tr_cost
    m_controller.objective = pyo.Objective(
        expr=sum(
            m_controller.tracking_cost[i, t]
            for i in m_controller.setpoint_set
            for t in m_controller.time
            if t != m_controller.time.first()
        )
    )

    #
    # Unfix input in controller model
    #
    m_controller.Qr[:].unfix()
    m_controller.Rec[:].unfix()
    m_controller.Qr[t0_controller].fix()
    m_controller.Rec[t0_controller].fix()
    sample_points = [i * sample_time for i in range(samples_per_controller_horizon + 1)]
    input_set, pwc_con = controller_interface.get_piecewise_constant_constraints(
        [m_controller.Qr, m_controller.Rec], sample_points
    )
    m_controller.input_set = input_set
    m_controller.pwc_con = pwc_con

    #
    # Construct sample-point set for measurements and model disturbances
    # in the estimator model
    #
    sample_points = [
        t0_estimator +
        sample_time*i for i in range(samples_per_estimator_horizon+1)
    ]
    m_estimator.sample_points = ContinuousSet(initialize=sample_points)

    measured_variables = \
        [pyo.Reference(m_estimator.T[:, i]) for i in m_estimator.tray] + \
        [pyo.Reference(m_estimator.Mv[:, i]) for i in m_estimator.tray]

    meas_set, measurements = get_parameters_from_variables(
        measured_variables, m_estimator.sample_points, ctype=pyo.Var
    )
    m_estimator.measurement_set = meas_set
    m_estimator.measurements = measurements
    m_estimator.measurements.fix()

    #
    # Construct disturbed model constraints
    #
    relaxed_constraints = \
        [m_estimator.de_M[:, idx] for idx in m_estimator.tray] + \
        [m_estimator.de_x[:, idx] for idx in m_estimator.tray]
    weight_data = {
        **{m_estimator.de_M[:, idx]: 1.0 for idx in m_estimator.tray},
        **{m_estimator.de_x[:, idx]: 1.0E4 for idx in m_estimator.tray},
    }
    n_con = len(relaxed_constraints)
    m_estimator.disturbance_set = pyo.Set(initialize=range(n_con))
    m_estimator.disturbance = pyo.Var(
        m_estimator.disturbance_set, m_estimator.time, initialize=0.0,
    )
    dist_con, expr = get_disturbed_constraint_and_residual_expression(
        m_estimator.time, relaxed_constraints,
        m_estimator.disturbance_set, m_estimator.disturbance,
        weight_data=weight_data,
    )
    m_estimator.disturbed_constraint = dist_con
    m_estimator.weighted_residual_expr = expr

    _, pwc_con = estimator_interface.get_piecewise_constant_constraints(
        slice_components(m_estimator.time, m_estimator.disturbance),
        sample_points,
    )
    m_estimator.piecewise_constant_residual_constraints = pwc_con
    # Deactivate original differential equations:
    for con in relaxed_constraints:
        con.deactivate()

    #
    # Make interface w.r.t. sample points
    #
    estimator_spt_interface = mpc.DynamicModelInterface(
        m_estimator, m_estimator.sample_points
    )

    #
    # Construct least square objective to minimize measurement errors
    # and model disturbances
    #
    error_weight = mpc.ScalarData(
        {**{m_estimator.T[:, i]: 10.0 for i in m_estimator.tray},
         **{m_estimator.Mv[:,i]: 1.0E6 for i in m_estimator.tray}}
    )
    _, cost = get_tracking_cost_from_target_trajectory(
        m_estimator.sample_points,
        m_estimator.measurements,
        variables=measured_variables,
        weight_data=error_weight,
    )
    m_estimator.measurement_error_cost = cost

    m_estimator.squred_error_disturbance_objective = pyo.Objective(expr=(
        sum(m_estimator.measurement_error_cost.values())
        + sum(
            m_estimator.weighted_residual_expr[i, t]
            for i in m_estimator.disturbance_set
            for t in sample_points
        )
    ))

    # Make the degree of freedom correct
    pyo.Reference(m_estimator.M[t0_estimator, :]).unfix()
    pyo.Reference(m_estimator.x[t0_estimator, :]).unfix()
    pyo.Reference(m_estimator.disturbance[:, t0_estimator]).fix(0.0)

    #
    # Set up a model linker to send measurements to estimator to update
    # measurement variables
    #
    measured_variables_in_plant = [m_plant.find_component(var.referent)
                                   for var in measured_variables
    ]
    flatten_measurements = [
        pyo.Reference(m_estimator.measurements[idx, :])
        for idx in m_estimator.measurement_set
    ]
    measurement_linker = DynamicVarLinker(
        measured_variables_in_plant,
        flatten_measurements,
    )

    #
    # Set up a model linker to send measurements to estimator to initialize
    # measured variables
    #
    estimate_linker = DynamicVarLinker(
        measured_variables_in_plant,
        measured_variables,
    )

    sim_t0 = 0.0

    #
    # Initialize data structure to hold results of "rolling horizon"
    # simulation.
    #
    sim_data = plant_interface.get_data_at_time([sim_t0])

    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller
    for i in range(simulation_steps):
        print("")
        print("Current cycle: ", i)

        # The starting point of this part of the simulation
        # in "real" time (rather than the model's time set)
        sim_t0 = i * sample_time
        sim_tf = (i + 1)*sample_time

        #
        # Solve controller model to get inputs
        #
        res = solver.solve(m_controller, tee=tee)
        pyo.assert_optimal_termination(res)
        ts_data = controller_interface.get_data_at_time(ts)
        input_data = ts_data.extract_variables(
            [m_controller.Qr, m_controller.Rec]
        )

        plant_interface.load_data(input_data)

        #
        # Solve plant model to simulate
        #
        res = solver.solve(m_plant, tee=tee)
        pyo.assert_optimal_termination(res)

        #
        # Extract data from simulated model
        #
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0 - m_plant.time.first())
        sim_data.concatenate(m_data)

        #
        # Load measurements from plant to estimator
        #
        tf_plant = m_plant.time.last()
        tf_estimator = m_estimator.time.last()
        measurement_linker.transfer(tf_plant, tf_estimator)

        #
        # Initialize measured variables within the last sample time to
        # current measurements
        #
        ncp = m_estimator.time.get_discretization_info()["ncp"]
        last_sample_time = list(m_estimator.time)[-ncp*ntfe_per_sample_estimator:]
        estimate_linker.transfer(tf_plant, last_sample_time)

        #
        # Load inputs into estimator
        #
        estimator_interface.load_data(
            input_data, last_sample_time
        )

        #
        # Solve estimator model to get estimates
        #
        res = solver.solve(m_estimator, tee=tee)
        pyo.assert_optimal_termination(res)

        #
        # Extract estimate data from estimator
        #
        estimator_data = estimator_interface.get_data_at_time(tf_estimator)
        #
        # Re-initialize plant model
        #
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)

        #
        # Re-initialize controller model
        #
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(estimator_data, time_points=t0_controller)

        #
        # Re-initialize estimator model
        #
        estimator_interface.shift_values_by_time(sample_time)
        estimator_spt_interface.shift_values_by_time(sample_time)

    return m_plant, sim_data


def main():
    init_steady_target = mpc.ScalarData({"T[*,14]": 355.0, "T[*,29]":348.3})
    init_data = get_steady_state_data(init_steady_target, tee=True)
    setpoint_target = mpc.ScalarData({"T[*,14]": 355.1, "T[*,29]":348.6})
    setpoint_data = get_steady_state_data(setpoint_target, tee=True)

    m, sim_data = run_cstr_mpc_mhe(
        init_data, setpoint_data, simulation_steps=100, tee=True
    )

    _plot_time_indexed_variables(sim_data, [m.T[:, 14], m.T[:, 29]], show=True)
    _step_time_indexed_variables(sim_data, [m.Rec[:]], show=True)
    _step_time_indexed_variables(sim_data, [m.Qr[:]], show=True)


if __name__ == "__main__":
    main()
