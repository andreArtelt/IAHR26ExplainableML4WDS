import random
import numpy as np
from epyt_flow.simulation import ScenarioConfig, ScenarioSimulator, SENSOR_TYPE_NODE_PRESSURE, EpanetConstants
from epyt_flow.utils import to_seconds, time_points_to_one_hot_encoding
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation.events import SensorFaultGaussian
from epyt_flow.visualization import ScenarioVisualizer


pressure_sensor_locations = ["5", "10", "16", "22", "30"]


def generate_data(scenarios_id: list[str], apply_sensor_faults: bool = True, apply_leakages: bool = False):
    X, y = [], []
    hyd_sim_data_plain = []

    network_configs = load_leakdb_scenarios(scenarios_id=scenarios_id,
                                            use_net1=False)
    for s_config in network_configs:
        # Remove leakages and other events
        s_config = ScenarioConfig(f_inp_in=s_config.f_inp_in,
                                  general_params=s_config.general_params,
                                  sensor_config=s_config.sensor_config)

        # Create scenario
        with ScenarioSimulator(f_inp_in=s_config.f_inp_in) as scenario:
            # Set simulation duration to 21 days
            scenario.set_general_parameters(simulation_duration=to_seconds(days=21))

            # Make sure the "right units" are used (important for the surrogate model!) -- i.e., m^3/s for flows and m for pressure
            scenario.set_general_parameters(flow_units_id=EpanetConstants.EN_CMS,
                                            pressure_units_id=EpanetConstants.EN_METERS)

            # Place pressure sensors at nodes "13", "16", "22", and "30"
            scenario.set_pressure_sensors(sensor_locations=pressure_sensor_locations)
            #scenario.place_pressure_sensors_everywhere()

            # Extract the location of the reservoirs -- recall that we need the heads at
            # all reservoirs as inputs for the hydraulic surrogate model
            reservoir_idx = [idx - 1 for idx in scenario.epanet_api.get_all_reservoirs_idx()]  # NOTE: EPANET starts counting at 1!

            # Add some random sensor faults and leakages
            if apply_sensor_faults is True:
                #n_gaussian_noise_faults = random.randint(2, 4)
                n_gaussian_noise_faults = 1
                for faulty_sensor_id in random.sample(scenario.sensor_config.pressure_sensors,
                                                      k=n_gaussian_noise_faults):
                    print(f"Add sensor fault at node {faulty_sensor_id}")
                    fault_std = random.uniform(5, 10)
                    start_day = random.randint(1, 20)
                    end_day = start_day + random.randint(1, 21-start_day)
                    print(f"Std of Gaussian noise added to faulty sensor: {fault_std}")
                    scenario.add_sensor_fault(SensorFaultGaussian(sensor_id=faulty_sensor_id,
                                                                  std=fault_std,
                                                                  sensor_type=SENSOR_TYPE_NODE_PRESSURE,
                                                                  start_time=to_seconds(days=start_day),
                                                                  end_time=to_seconds(days=end_day)))
   
            if apply_leakages is True:
                #n_leaks = random.randint(2, 5)
                n_leaks = 1
                for leaky_node_id in random.sample(scenario.sensor_config.nodes, k=n_leaks):
                    start_day = random.randint(5, 20)
                    end_day = start_day + random.randint(1, 21-start_day)

                    start_idx = to_seconds(days=start_day) // scenario.get_hydraulic_time_step()
                    end_idx = to_seconds(days=end_day) // scenario.get_hydraulic_time_step()
                    print(f"Leakage at node {leaky_node_id} -- start: {start_idx} end: {end_idx}")

                    nodal_base_demand = scenario.get_node_base_demand(leaky_node_id)
                    nodal_demands = scenario.get_node_demand_pattern(leaky_node_id)
                    nodal_demands[start_idx:end_idx] *= 100    # Increase demands to simulate a leackage

                    node_idx = scenario.epanet_api.get_node_idx(leaky_node_id)
                    node_pattern_idx = int(scenario.epanet_api.getnodevalue(node_idx, EpanetConstants.EN_PATTERN))
                    demand_pat_id = scenario.epanet_api.getpatternid(node_pattern_idx)

                    scenario.set_node_demand_pattern(leaky_node_id, nodal_base_demand, demand_pat_id, nodal_demands)

            # Run simulation
            scada_data = scenario.run_simulation()

            X_ = scada_data.get_data()

            events_times = [int(t / scenario.get_hydraulic_time_step())
                            for t in scenario.get_events_active_time_points()]
            y_ = time_points_to_one_hot_encoding(events_times, total_length=X_.shape[0])

            X.append(X_)
            y.append(y_)

            # Extract data, necessary to run the hydraulic surrogate model
            # Heads(m) = pressure(m) + elevation(m)
            scenario.place_demand_sensors_everywhere()
            scenario.place_pressure_sensors_everywhere()
            scenario.place_flow_sensors_everywhere()

            scada_data = scenario.run_simulation()

            topo = scada_data.network_topo
            demands = scada_data.get_data_demands()
            pressures = scada_data.get_data_pressures()
            flows = scada_data.get_data_flows()

            elevs_np = np.array([float(topo.get_node_info(node_id).get("elevation", 0.0))
                                for node_id in topo.get_all_nodes()], dtype=np.float32)
            elevs_tile = elevs_np[None, :].repeat(pressures.shape[0], axis=0)

            heads = (pressures[:, :len(topo.get_all_nodes())] + elevs_tile[:, :len(topo.get_all_nodes())])

            hyd_sim_data_plain.append((heads, reservoir_idx, demands, flows))

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), hyd_sim_data_plain
