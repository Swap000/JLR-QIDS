import gym
from environment import VehicularEnv
from agents import (
    BrakingAgent, EngineAgent, SteeringAgent,
    InfotainmentAgent, CommunicationAgent, CoordinatorAgent
)
from utils import preprocess_state, preprocess_batch
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = VehicularEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = {
        'braking': BrakingAgent(state_size, action_size),
        'engine': EngineAgent(state_size, action_size),
        'steering': SteeringAgent(state_size, action_size),
        'infotainment': InfotainmentAgent(state_size, action_size),
        'communication': CommunicationAgent(state_size, action_size),
        'coordinator': CoordinatorAgent(state_size, action_size)
    }

    episodes = 1000
    rewards = []

    for e in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = preprocess_state(state)
            actions = {name: agent.quantum_act(state) for name, agent in agents.items() if name != 'coordinator'}
            aggregated_state = torch.cat([torch.tensor(state) for state in actions.values()])

            action_coordinator = agents['coordinator'].quantum_act(aggregated_state)

            next_state, reward, done, _ = env.step(action_coordinator)
            episode_reward += reward

            for name, agent in agents.items():
                if name != 'coordinator':
                    agent.train([state], [actions[name]], [reward], [next_state], [done])
                else:
                    agent.train([aggregated_state], [action_coordinator], [reward], [next_state], [done])

            state = next_state

        rewards.append(episode_reward)

        if e % 100 == 0:
            print(f"Episode {e} finished with reward {episode_reward}")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.show()

if __name__ == "__main__":
    main()