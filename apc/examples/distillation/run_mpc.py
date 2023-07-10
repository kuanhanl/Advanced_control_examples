import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from apc.examples.distillation.model import (
    create_instance,
)
from pyomo.contrib.mpc.examples.cstr.model import (
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)


def get_steady_state_data(target, tee=False):
    m = create_instance(dynamic=False)
    interface = mpc.DynamicModelInterface(m, m.time)
    var_set, tr_cost = interface.get_penalty_from_target(target)
    m.target_set = var_set
    m.tracking_cost = tr_cost
    m.objective = pyo.Objective(expr=sum(m.tracking_cost[:, 0]))
    m.Qr[:].unfix()
    m.Rec[:].unfix()
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=tee)
    return interface.get_data_at_time(0)


def run_dist_mpc(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=20,
    sample_time=60.0,
    ntfe_per_sample_controller=1,
    ntfe_plant=3,
    simulation_steps=50,
    tee=False,
):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.time)
    t0_controller = m_controller.time.first()

    m_plant = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)

    # Sets initial conditions and initializes
    controller_interface.load_data(initial_data)
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
        # Re-initialize plant model
        #
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)

        #
        # Re-initialize controller model
        #
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)

    return m_plant, sim_data


def main():
    init_steady_target = mpc.ScalarData({"T[*,14]": 355.0, "T[*,29]":348.3})
    init_data = get_steady_state_data(init_steady_target, tee=True)
    setpoint_target = mpc.ScalarData({"T[*,14]": 355.1, "T[*,29]":348.6})
    setpoint_data = get_steady_state_data(setpoint_target, tee=True)

    m, sim_data = run_dist_mpc(
        init_data, setpoint_data, simulation_steps=10, tee=True
    )

    _plot_time_indexed_variables(sim_data, [m.T[:, 14], m.T[:, 29]], show=True)
    _step_time_indexed_variables(sim_data, [m.Rec[:]], show=True)
    _step_time_indexed_variables(sim_data, [m.Qr[:]], show=True)


if __name__ == "__main__":
    main()