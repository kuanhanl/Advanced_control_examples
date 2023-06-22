import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

from pyomo.dae import ContinuousSet
from pyomo.contrib.mpc.examples.cstr.model import (
    create_instance,
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_time_varying_target,
)
from apc.mhe_utils import (
    get_parameters_from_variables,
    get_constraint_residual_expression,
    slice_components,
    get_tracking_cost_from_target_trajectory,
)
from run_mhe import plot_states_estimates_from_data


def run_cstr_mpc_mhe(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=5,
    samples_per_estimator_horizon=5,
    sample_time=2.0,
    ntfe_per_sample_controller=2,
    ntfe_per_sample_estimator=5,
    ntfe_plant=5,
    simulation_steps=5,
    tee=False,
):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.time)
    t0_controller = m_controller.time.first()

    estimator_horizon = sample_time * samples_per_estimator_horizon
    ntfe = ntfe_per_sample_estimator * samples_per_estimator_horizon
    m_estimator = create_instance(horizon=estimator_horizon, ntfe=ntfe)
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
    setpoint_variables = [m_controller.conc[:, "A"], m_controller.conc[:, "B"]]
    vset, tr_cost = controller_interface.get_penalty_from_target(
        setpoint_data, variables=setpoint_variables
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
    m_controller.flow_in[:].unfix()
    m_controller.flow_in[t0_controller].fix()
    sample_points = [i * sample_time for i in range(samples_per_controller_horizon + 1)]
    input_set, pwc_con = controller_interface.get_piecewise_constant_constraints(
        [m_controller.flow_in], sample_points
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

    measured_variables = [pyo.Reference(m_estimator.conc[:, "A"])]

    meas_set, measurements = get_parameters_from_variables(
        measured_variables, m_estimator.sample_points, ctype=pyo.Var
    )
    m_estimator.measurement_set = meas_set
    m_estimator.measurements = measurements
    m_estimator.measurements.fix()

    #
    # Construct disturbed model constraints
    #
    relaxed_constraints = [
        m_estimator.conc_diff_eqn[:, idx] for idx in m_estimator.comp
    ]
    weight_data = {key: 10.0 for key in relaxed_constraints}

    resid_set, resid = get_constraint_residual_expression(
        relaxed_constraints, m_estimator.time, weight_data=weight_data,
    )
    m_estimator.disturbance_set = resid_set
    m_estimator.residual_expr = resid
    _, pwc_con = estimator_interface.get_piecewise_constant_constraints(
        slice_components(m_estimator.time, m_estimator.residual_expr),
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
    _, cost = get_tracking_cost_from_target_trajectory(
        m_estimator.sample_points,
        m_estimator.measurements,
        variables=measured_variables,
    )
    m_estimator.measurement_error_cost = cost

    m_estimator.squred_error_disturbance_objective = pyo.Objective(expr=(
        sum(m_estimator.measurement_error_cost.values())
        + sum(
            m_estimator.residual_expr[i, t]
            for i in m_estimator.disturbance_set
            for t in sample_points
        )
    ))

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
    estimate_data = estimator_interface.get_data_at_time([sim_t0])

    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller
    for i in range(simulation_steps):
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
        input_data = ts_data.extract_variables([m_controller.flow_in])

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
        last_sample_time = list(m_estimator.time)[-ntfe_per_sample_estimator:]
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
        estimator_data = estimator_interface.get_data_at_time([tf_estimator])
        # Shift time points from "estimator time" to "simulation time"
        estimator_data.shift_time_points(sim_tf-tf_estimator)
        estimate_data.concatenate(estimator_data)

        #
        # Re-initialize plant model
        #
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)

        #
        # Re-initialize controller model
        #
        controller_interface.shift_values_by_time(sample_time)
        estimator_data.shift_time_points(-sim_tf)
        controller_interface.load_data(estimator_data, time_points=[t0_controller])

        #
        # Re-initialize estimator model
        #
        estimator_interface.shift_values_by_time(sample_time)
        estimator_spt_interface.shift_values_by_time(sample_time)

    return m_plant, sim_data, estimate_data


def main():
    init_steady_target = mpc.ScalarData({"flow_in[*]": 0.3})
    init_data = get_steady_state_data(init_steady_target, tee=False)
    setpoint_target = mpc.ScalarData({"flow_in[*]": 1.2})
    setpoint_data = get_steady_state_data(setpoint_target, tee=False)

    m, sim_data, estimate_data = run_cstr_mpc_mhe(
        init_data, setpoint_data, tee=False
    )

    _plot_time_indexed_variables(sim_data, [m.conc[:, "A"], m.conc[:, "B"]], show=True)
    _step_time_indexed_variables(sim_data, [m.flow_in[:]], show=True)

    plot_states_estimates_from_data(
        sim_data,
        estimate_data,
        [m.conc[:, "A"], m.conc[:, "B"]],
        show=True,
    )


if __name__ == "__main__":
    main()

