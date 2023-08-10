import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
    create_instance,
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data

from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface as SensInt
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface

import time
from apc.plot_CPU_time import _plot_CPU_time


def run_cstr_asmpc(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=5,
    sample_time=2.0,
    ntfe_per_sample_controller=2,
    ntfe_plant=5,
    simulation_steps=5,
    tee=False,
):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.time)
    t0_controller = m_controller.time.first()

    m_predictor = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    predictor_interface = mpc.DynamicModelInterface(m_predictor, m_predictor.time)
    
    m_plant = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)

    # Sets initial conditions and initializes
    controller_interface.load_data(initial_data)
    predictor_interface.load_data(initial_data)
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

    sim_t0 = 0.0

    #
    # Initialize data structure to hold results of "rolling horizon"
    # simulation.
    #
    sim_data = plant_interface.get_data_at_time([sim_t0])

    #
    # Set up NLP sensitivity solvers
    #
    k_aug = pyo.SolverFactory('k_aug', solver_io='nl')
    k_aug.options['dsdp_mode'] = "" # Sensitivity mode
    dot_sens = pyo.SolverFactory('dot_sens', solver_io='nl')
    dot_sens.options["dsdp_mode"] = "" # Sensitivity mode

    #
    # Set up k_aug interface
    #
    k_aug_interface = K_augInterface(k_aug=k_aug, dot_sens=dot_sens)
    sens = SensInt(m_controller, clone_model=False)

    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller
    online_CPU = []
    offline_CPU = []
    for i in range(simulation_steps):
        # The starting point of this part of the simulation
        # in "real" time (rather than the model's time set)
        sim_t0 = i * sample_time

        #
        # Solve for inputs
        #
        if i == 0:
            # First control problem has sufficient time to be solved
            res = solver.solve(m_controller, tee=tee)
            pyo.assert_optimal_termination(res)
        else:
            # Online sensitivity update 
            online_start = time.time()
            sens.perturb_parameters(true_init_vals)
            k_aug_interface.dot_sens(m_controller, tee=tee)
            online_end = time.time()
            online_CPU.append(online_end-online_start)
            
        ts_data = controller_interface.get_data_at_time(ts)
        input_data = ts_data.extract_variables([m_controller.flow_in])

        predictor_interface.load_data(input_data)

        #
        # Solve predictor model to predict states
        #
        res = solver.solve(m_predictor, tee=tee)
        pyo.assert_optimal_termination(res)

        pred_data = predictor_interface.get_data_at_time(m_predictor.time.last())

        #
        # Re-initialize controller model
        #
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(pred_data, time_points=t0_controller)
        
        #
        # Declare initial states as pertuted parameters
        #
        pred_init = [
            m_controller.conc[0, "A"], m_controller.conc[0, "B"]
        ]
        sens.setup_sensitivity(pred_init) # Suffixes are constructed here!
        
        offline_start = time.time()
        
        #
        # Solve controller model one step ahead with predictive states
        # 
        res = solver.solve(m_controller, tee=tee)
        pyo.assert_optimal_termination(res)
        
        #
        # Solve for the sensitivity matrix
        #
        m_controller.ipopt_zL_in.update(m_controller.ipopt_zL_out)
        m_controller.ipopt_zU_in.update(m_controller.ipopt_zU_out)
        k_aug_interface.k_aug(m_controller, tee=tee)

        plant_interface.load_data(input_data)

        #
        # Solve plant model to simulate
        #
        res = solver.solve(m_plant, tee=tee)
        pyo.assert_optimal_termination(res)
        # Add noise here
        
        #
        # Save true plant state
        #
        tf = m_plant.time.last()
        true_init_vals = [
            m_plant.conc[tf, "A"].value, m_plant.conc[tf, "B"].value
        ]

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
        # Re-initialize predictor model
        #
        predictor_interface.load_data(tf_data)

        offline_end = time.time()
        offline_CPU.append(offline_end-offline_start)
        
    return m_plant, sim_data, online_CPU, offline_CPU


def main():
    init_steady_target = mpc.ScalarData({"flow_in[*]": 0.3})
    init_data = get_steady_state_data(init_steady_target, tee=False)
    setpoint_target = mpc.ScalarData({"flow_in[*]": 1.2})
    setpoint_data = get_steady_state_data(setpoint_target, tee=False)

    m, sim_data, online_CPU, offline_CPU = run_cstr_asmpc(init_data, setpoint_data, tee=False)

    _plot_time_indexed_variables(sim_data, [m.conc[:, "A"], m.conc[:, "B"]], show=True)
    _step_time_indexed_variables(sim_data, [m.flow_in[:]], show=True)
    
    _plot_CPU_time(online_CPU, offline_CPU)


if __name__ == "__main__":
    main()