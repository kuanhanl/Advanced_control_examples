import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

from pyomo.dae import ContinuousSet
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
from pyomo.contrib.mpc.examples.cstr.model import create_instance
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_time_varying_target,
)
from apc.mhe_utils import (
    get_parameters_from_variables,
    get_constraint_residual_expression,
    slice_components,
    get_tracking_cost_from_target_trajectory,
)


def get_control_inputs(sample_time=2.0):
    n_samples = 5
    control_input_time = [i*sample_time for i in range(n_samples)]
    control_input_data = {
        'flow_in[*]': [0.4 + 0.1*i for i in range(n_samples)]
    }
    series = mpc.TimeSeriesData(control_input_data, control_input_time)
    # Note that if we want a json representation of this data, we
    # can always call json.dump(fp, series.to_serializable()).
    return series


def run_cstr_mhe(
    initial_data,
    samples_per_estimator_horizon=5,
    sample_time=2.0,
    ntfe_per_sample_estimator=5,
    ntfe_plant=5,
    simulation_steps=5,
    tee=False,
):
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
        last_sample_time = list(m_estimator.time)[-ntfe_per_sample_estimator:]
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


def plot_states_estimates_from_data(
        state_data,
        estimate_data,
        names,
        show=False,
        save=False,
        fname=False,
        transparent=False
        ):
    state_time = state_data.get_time_points()
    states = state_data.get_data()
    estimate_time = estimate_data.get_time_points()
    estimates = estimate_data.get_data()
    from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
    cuids = [get_indexed_cuid(name) for name in names]

    import matplotlib.pyplot as plt
    for i, cuid in enumerate(cuids):
        fig, ax = plt.subplots()
        state_values = states[cuid]
        estimate_values = estimates[cuid]
        ax.plot(state_time, state_values, label="Plant states")
        ax.plot(estimate_time, estimate_values, "o", label="Estimates")
        ax.set_title(cuid)
        ax.set_xlabel("Time")
        ax.legend()

        if show:
            plt.show()
        if save:
            if fname is None:
                fname = "state_estimate%s.png" % i
            fig.savefig(fname, transparent=transparent)


def main():
    init_steady_target = mpc.ScalarData({"flow_in[*]": 0.3})
    init_data = get_steady_state_data(init_steady_target, tee=False)

    m, sim_data, estimate_data = run_cstr_mhe(init_data, tee=True)

    plot_states_estimates_from_data(
        sim_data,
        estimate_data,
        [m.conc[:, "A"], m.conc[:, "B"]],
        show=True,
    )


if __name__ == "__main__":
    main()
