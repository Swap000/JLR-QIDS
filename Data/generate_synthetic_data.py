import os
import json
import numpy as np
from environment.vehicular_env import VehicularEnv

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def generate_data_for_agent(agent_name, env, num_samples, output_dir):
    data = []
    scenarios = ["normal_driving", "sudden_braking", "malicious_signal", "emergency", "erratic_driving","weather_condition", "traffic_condition",
"signal_spoofing", "jamming","denial_of_service", "data_injection", "man_in_the_middle", "replay_attack", "malware_infection", "fuzzing"]

    for _ in range(num_samples):
        scenario = np.random.choice(scenarios)
        data.extend(env.simulate_scenario(scenario))

    file_path = os.path.join(output_dir, f'{agent_name}_data.json')
    save_data(data, file_path)
    print(f'Saved {num_samples} samples of synthetic data for {agent_name} to {file_path}')

def main():
    num_samples = 5000000  # Adjust this for the data size
    output_dir = './synthetic_data'
    os.makedirs(output_dir, exist_ok=True)

    env = VehicularEnv()
    agents = ["braking_system_agent", "ecu_agent", "steering_system_agent", "infotainment_system_agent", "communication_system_agent"]

    for agent in agents:
        generate_data_for_agent(agent, env, num_samples, output_dir)

if __name__ == '__main__':
    main()