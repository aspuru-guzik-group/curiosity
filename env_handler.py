import numpy as np
import torch
import helper

def sample_env_trajectories_parallel(env, agent, batch_size, device, i_episode=0):        
    helper.reset_all_recurrent_modules(agent, batch_size=batch_size)

    env.reset()
    dones = np.array([False])
    initial_state = None
    last_idx_list = None
    states = []
    last_idx_list_all = []
    idx_batch = []
    actions = []
    rewards = []
    action_probs_unmasked = []
    action_probs_masked = []
    idx_probs_unmasked = []
    idx_probs_masked = []
    encoded_states = [] 
    state_values = []
    mol_properties = []
    dones_list = []
    curiosity_errors = []
    while not np.all(dones):
        state, last_idx_list, action, idx_probs_unmasked_batch, idx_probs_masked_batch, action_probs_unmasked_batch, action_probs_masked_batch, encoded_state, state_value, reward, next_state, dones, mol_property, curiosity_error = do_step_parallel(env, agent, initial_state=initial_state, last_idx_list=last_idx_list, i_episode=i_episode)
        idx, action = action

        states.append(state)
        last_idx_list_all.append(last_idx_list)
        idx_batch.append(idx)
        actions.append(action)
        rewards.append(reward)

        action_probs_unmasked.append(action_probs_unmasked_batch)
        action_probs_masked.append(action_probs_masked_batch)
        idx_probs_unmasked.append(idx_probs_unmasked_batch)
        idx_probs_masked.append(idx_probs_masked_batch)
        encoded_states.append(encoded_state)
        state_values.append(state_value)
        initial_state = next_state
        mol_properties.append(mol_property)
        dones_list.append(torch.tensor(dones))
        curiosity_errors.append(curiosity_error)


    return states, last_idx_list_all, idx_batch, actions, idx_probs_unmasked, idx_probs_masked, action_probs_unmasked, action_probs_masked, encoded_states, state_values, rewards, mol_properties, dones_list, np.array(curiosity_errors).mean()


def do_step_parallel(env, agent, initial_state=None, last_idx_list=None, i_episode=0):
    if initial_state is None:
        state, last_idx_list = env.reset()
        state = state.float()

    else:
        state = initial_state

    action, idx_probs_unmasked, idx_probs_masked, action_probs_unmasked, action_probs_masked, encoded_state, state_value = agent.select_action(state, last_idx_list)

    next_state, last_idx_list, reward, done, _, mol_property, curiosity_error = env.step(action, i_episode=i_episode)

    return state, last_idx_list, action, idx_probs_unmasked, idx_probs_masked, action_probs_unmasked, action_probs_masked, encoded_state, state_value, reward, next_state, done, mol_property, curiosity_error
