import random
import numpy as np
from epyt_flow.simulation import ScenarioConfig, ScenarioSimulator, SENSOR_TYPE_NODE_PRESSURE
from epyt_flow.utils import to_seconds, time_points_to_one_hot_encoding
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation.events import SensorFaultGaussian, AbruptLeakage
from epyt_flow.visualization import ScenarioVisualizer


pressure_sensor_locations = ["5", "10", "16", "22", "30"]


def generate_data(scenarios_id: list[str], apply_sensor_faults: bool = True, apply_leakages: bool = False):
    X, y = [], []

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

            # Place pressure sensors at nodes "13", "16", "22", and "30"
            scenario.set_pressure_sensors(sensor_locations=pressure_sensor_locations)
            #scenario.place_pressure_sensors_everywhere()

            #scenario.plot_topology()
            vis = ScenarioVisualizer(scenario)
            vis.highlight_sensor_config()
            #vis.add_labels('sensor_config', font_size=10)
            vis.show_plot(export_to_file="Hanoi-topo.png", suppress_plot=True)

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
                    scenario.add_sensor_fault(SensorFaultGaussian(sensor_id=faulty_sensor_id,
                                                                  std=fault_std,
                                                                  sensor_type=SENSOR_TYPE_NODE_PRESSURE,
                                                                  start_time=to_seconds(days=start_day),
                                                                  end_time=to_seconds(days=end_day)))
                    
            if apply_leakages is True:
                #n_leaks = random.randint(2, 5)
                n_leaks = 1
                for leaky_link_id in random.sample(scenario.sensor_config.links, k=n_leaks):
                    start_day = random.randint(1, 20)
                    end_day = start_day + random.randint(1, 21-start_day)
                    leak = AbruptLeakage(link_id=leaky_link_id, node_id=None,
                                         diameter=0.5,
                                         start_time=to_seconds(days=start_day),
                                         end_time=to_seconds(days=end_day))
                    scenario.add_leakage(leak)

            # Run simulation
            scada_data = scenario.run_simulation()

            X_ = scada_data.get_data()

            events_times = [int(t / scenario.get_hydraulic_time_step())
                            for t in scenario.get_events_active_time_points()]
            y_ = time_points_to_one_hot_encoding(events_times, total_length=X_.shape[0])

            X.append(X_)
            y.append(y_)

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)
