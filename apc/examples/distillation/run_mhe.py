import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

from pyomo.dae import ContinuousSet
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_time_varying_target,
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
from apc.examples.cstr.run_mhe import plot_states_estimates_from_data


def get_control_inputs(sample_time=60.0):
    n_samples = 5
    control_input_time = [i*sample_time for i in range(n_samples)]
    control_input_data = {
        'Qr[*]': [1.64E6 - 1.0E4*i for i in range(1, n_samples+1)],
        'Rec[*]': [1.20 + 0.1*i for i in range(1, n_samples+1)]
    }
    series = mpc.TimeSeriesData(control_input_data, control_input_time)
    # Note that if we want a json representation of this data, we
    # can always call json.dump(fp, series.to_serializable()).
    return series


def run_cstr_mhe(
    initial_data,
    samples_per_estimator_horizon=10,
    sample_time=60.0,
    ntfe_per_sample_estimator=2,
    ntfe_plant=2,
    simulation_steps=5,
    tee=False,
):
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
    estimator_interface.load_data(initial_data)
    plant_interface.load_data(initial_data)

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

    #
    # Load control input data for simulation
    #
    control_inputs = get_control_inputs()

    sim_t0 = 0.0

    #
    # Initialize data structure to hold results of "rolling horizon"
    # simulation.
    #
    sim_data = plant_interface.get_data_at_time([sim_t0])
    estimate_data = estimator_interface.get_data_at_time([sim_t0])


    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_estimator
    for i in range(simulation_steps):
        print("")
        print("Current cycle: ", i)

        # The starting point of this part of the simulation
        # in "real" time (rather than the model's time set)
        sim_t0 = i*sample_time
        sim_tf = (i + 1)*sample_time

        #
        # Load inputs into plant
        #
        current_control = control_inputs.get_data_at_time(time=sim_t0)
        plant_interface.load_data(
            current_control, non_initial_plant_time
        )

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
            current_control, last_sample_time
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
        # Re-initialize estimator model
        #
        estimator_interface.shift_values_by_time(sample_time)
        estimator_spt_interface.shift_values_by_time(sample_time)

        #
        # Re-initialize plant model to final values.
        # This sets new initial conditions, including inputs.
        #
        plant_interface.copy_values_at_time(source_time=tf_plant)

    return m_plant, sim_data, estimate_data


def main():
    init_steady_target = mpc.ScalarData({"T[*,14]": 355.0, "T[*,29]":348.3})
    init_data = get_steady_state_data(init_steady_target, tee=True)

    m, sim_data, estimate_data = run_cstr_mhe(init_data, tee=True)

    plot_states_estimates_from_data(
        sim_data,
        estimate_data,
        [m.x[:, 20], m.x[:, 40]],
        show=True,
    )


if __name__ == "__main__":
    main()
