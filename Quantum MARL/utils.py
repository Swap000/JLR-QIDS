import torch

def preprocess_state(state):
    return torch.FloatTensor(state)

def preprocess_batch(states, actions, rewards, next_states, dones):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    return states, actions, rewards, next_states, dones