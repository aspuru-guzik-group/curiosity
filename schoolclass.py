import os
from torch.distributions import Categorical
from neural_networks import *
import helper
import config


class BasicAgent(object):

    def __init__(self, lr_dict, device, environment, algorithm, identifier):
        super(BasicAgent, self).__init__()
        self.meta_data = dict()
        self.meta_data['lr_dict'] = lr_dict
        self.meta_data['identifier'] = identifier
        self.environment = environment
        self.algorithm = algorithm
        self.device = device
        self.save_directory_path = None

        optim_list = []
        for key in self.networks:
            self.networks[key].to(config.device)
            optim_list.append({'params' : self.networks[key].parameters(), 'lr': lr_dict[key]})

        self.optimizer = optim.Adam(optim_list, weight_decay=1e-3) 


        self.loss = 0
        self.base_path = './snapshots/{}'.format(self.environment)

        self.normalizer = helper.Welford()


    def _optimize(self):
        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        for key in self.networks:
            for param in self.networks[key].parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

        self.optimizer.step()



    def get_save_directory_path(self):
        if self.save_directory_path is None:
            experiment_list = os.listdir(self.base_path)
            experiment_list = [e.split('_')[1] for e in experiment_list if e.split('_')[0] == self.algorithm]
            
            
            if len(experiment_list) == 0:
                n_experiments = str(0)
            else:
                n_experiments = str(max([int(n) for n in experiment_list]) + 1)
        
        
            self.save_directory_path = self.base_path + '/' + self.algorithm + '_' + n_experiments
            if not os.path.exists(self.save_directory_path):
                os.mkdir(self.save_directory_path)
        
        return self.save_directory_path

    def snapshot_experiment(self, rewards, alternate_path=None, id=0):
        if alternate_path is None:    
            if not os.path.exists(self.base_path):
                os.makedirs(self.base_path)
            
            save_directory_path = self.get_save_directory_path()
        else:
            save_directory_path = alternate_path 
            self.save_directory_path = alternate_path

        torch.save(self, '{}/model_{}.pt'.format(save_directory_path, id))

        with open('{}/meta_{}.txt'.format(save_directory_path, id), 'w') as f:
            print(self.meta_data, file=f)
        
        np.savetxt('{}/rewards_{}.gzip'.format(save_directory_path, id), np.array(rewards))
     

    def calc_normalization(self, new_rewards):
        self.normalizer(new_rewards)

        return self.normalizer.mean, self.normalizer.std




class PPOAgent(BasicAgent):
    def __init__(self, networks, device, environment, lr = [1e-3, 1e-3, 1e-3], identifier = 'None', eps_clip = 0.2, K_epochs = 4, action_mode='APPEND', discount_factor=1, entropy_weight=0.1):

        self.networks = networks
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_weight = entropy_weight

        
        self.MseLoss = nn.MSELoss()

        self.sequence_like = helper.is_sequence_like(self)

        self.action_mode = action_mode

        self.discount_factor = discount_factor

        BasicAgent.__init__(self, lr, device, environment, 'PPO', identifier = identifier)


    def calculate_action_values(self, state):
        encoded_state = self.networks['state_encoder'](state)
        encoded_state_value_network = self.networks['state_encoder_value_network'](state)
        probs = self.networks['policy'](encoded_state)
        state_value = self.networks['critic'](encoded_state_value_network)
        
        return probs, encoded_state, state_value


    def mask_probs(self, probs, last_valid_action_idxs, idx_batch = None):
        if idx_batch is None:
            idx_probs_unmasked = probs

            idx_mask = torch.zeros_like(idx_probs_unmasked)

            
            for i in range(idx_probs_unmasked.shape[0]):
                idx_mask[i,:last_valid_action_idxs[i]+2] = 1
            
            
            idx_probs_masked = idx_probs_unmasked * idx_mask
            idx_probs_masked = idx_probs_masked / idx_probs_masked.sum().item()

            return idx_probs_masked

        else:
            action_probs_unmasked = probs

            
            modify_last_symbol = (idx_batch == last_valid_action_idxs)
            append_new_symbol = idx_batch == (last_valid_action_idxs + 1)
            no_emppty_strings1 = last_valid_action_idxs != torch.zeros_like(idx_batch)
            no_emppty_strings2 = last_valid_action_idxs != (torch.zeros_like(idx_batch) + 1)
            stop_symbol_mask = (modify_last_symbol | append_new_symbol) & (no_emppty_strings1 == no_emppty_strings2)
            
            
            action_probs_masked = action_probs_unmasked * 1
            
            action_probs_masked[:,-1] = action_probs_unmasked[:,-1] * stop_symbol_mask
            action_probs_masked = action_probs_masked / action_probs_masked.sum(-1)[:,None]
            

            return action_probs_masked
        
            
    def select_action(self, state_batch, last_valid_action_idxs):
        if isinstance(state_batch, list):
            state_batch = torch.nn.utils.rnn.pad_sequence(state_batch, batch_first=True)

        
        probs_unmasked, encoded_state, state_value = self.calculate_action_values(state_batch)
        idx_probs_unmasked, action_probs_unmasked = probs_unmasked

        idx_probs_unmasked = idx_probs_unmasked + 1e-7
        action_probs_unmasked = action_probs_unmasked + 1e-7
        idx_probs_unmasked = idx_probs_unmasked/ idx_probs_unmasked.sum(-1).unsqueeze(-1)
        action_probs_unmasked = action_probs_unmasked/ action_probs_unmasked.sum(-1).unsqueeze(-1)

        
        
        # idx_probs.shape = (batch, seq_length, max_length_string) Take only last valid time step using last_valid_action_idxs
        idx_array = torch.tensor(np.arange(idx_probs_unmasked.shape[0]))
        idx_probs_unmasked = idx_probs_unmasked[idx_array, last_valid_action_idxs]
        action_probs_unmasked = action_probs_unmasked[idx_array, last_valid_action_idxs]
        encoded_state = encoded_state[idx_array, last_valid_action_idxs]
        state_value = state_value[idx_array, last_valid_action_idxs]

        idx_probs_masked = self.mask_probs(idx_probs_unmasked, last_valid_action_idxs )
        m_idx = Categorical(idx_probs_masked)
        
        if self.action_mode == 'EDITING':
            idx = m_idx.sample() 
        elif self.action_mode == 'APPEND':
            if state_batch.sum() == 0:
                idx = torch.zeros_like(last_valid_action_idxs)
            else:
                idx = last_valid_action_idxs + 1

        
        action_probs_masked = self.mask_probs(action_probs_unmasked, last_valid_action_idxs, idx_batch=idx)
        m_action = Categorical(action_probs_masked)
        action = m_action.sample()

       
        return (idx,action), idx_probs_unmasked, idx_probs_masked, action_probs_unmasked, action_probs_masked, encoded_state, state_value


    def optimize(self, state_batch, last_idx_list_all, idx_batch, action_batch, idx_probs_on_policy, idx_probs_exploration, action_probs_on_policy, action_probs_exploration, encoded_states_batch, state_values_batch, reward_batch, dones_batch):
        # everything has shape seq_length, batch , 1/(seq_length_of_state, dim) at the moment

        # This has shape batch,seq_length and contains for every state how far we should go along the seq axis, only use for state batch
        last_idx_list_all_stacked = torch.stack(last_idx_list_all).transpose(1,0)
        # This has shape batch, and contains how far we should go along the seq axis, use for all but state batch
        last_idx_list_all_last_idx = last_idx_list_all_stacked[:,-1] + 1


        

        dones_batch = ~torch.stack(dones_batch).transpose(1,0)
        # use with [array_idx, :dones_batch_last_idx] or [array_idx, dones_batch_last_idx + 1] 
        dones_batch_last_idx = dones_batch.sum(-1) + 1
        

        batch_size = dones_batch.shape[0]
        
        
        # need to get shape batch, seq_length in list format to be able to use normalization
        reward_batch = np.array(reward_batch).swapaxes(1,0)
        reward_batch = [torch.tensor([reward_batch[i][j] for j in range(reward_batch.shape[1])]) for i in range(reward_batch.shape[0])]
        
        reward_batch = [helper.discount_rewards(rewards, self.discount_factor) for rewards in reward_batch]
        
        reward_batch = torch.stack(reward_batch)

       
        # calculate reward based only on states that are going to be in train batch
        flattened_rewards = [reward_batch[i][:dones_batch_last_idx[i]] for i in range(reward_batch.shape[0])]
        flattened_rewards = torch.cat(flattened_rewards)
        mean, std = self.calc_normalization(flattened_rewards.detach().cpu().numpy())  
        
        reward_batch = (reward_batch - mean) / (std + 1e-2)
        reward_batch = [torch.tensor([reward_batch[j][i] for j in range(reward_batch.shape[0])]) for i in range(reward_batch.shape[1])]
        reward_batch = torch.stack(reward_batch).transpose(1,0)
        reward_batch = [reward_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        reward_batch = torch.cat(reward_batch)
        #reward_batch = reward_batch[dones_batch]
        
        
        idx_probs_on_policy = torch.stack(idx_probs_on_policy).transpose(1,0)
        idx_probs_on_policy = [idx_probs_on_policy[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        idx_probs_on_policy = torch.cat(idx_probs_on_policy) 
        #idx_probs_on_policy = idx_probs_on_policy[dones_batch]
        
        idx_probs_exploration = torch.stack(idx_probs_exploration).transpose(1,0)
        idx_probs_exploration = [idx_probs_exploration[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        idx_probs_exploration = torch.cat(idx_probs_exploration)
        #idx_probs_exploration = idx_probs_exploration[dones_batch]
        
        action_probs_on_policy = torch.stack(action_probs_on_policy).transpose(1,0)
        action_probs_on_policy = [action_probs_on_policy[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        action_probs_on_policy = torch.cat(action_probs_on_policy) 
        #action_probs_on_policy = action_probs_on_policy[dones_batch]
        
        action_probs_exploration = torch.stack(action_probs_exploration).transpose(1,0)
        action_probs_exploration = [action_probs_exploration[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        action_probs_exploration = torch.cat(action_probs_exploration)
        #action_probs_exploration = action_probs_exploration[dones_batch]
        
        idx_batch = torch.stack(idx_batch).transpose(1,0)
        idx_batch = [idx_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        idx_batch = torch.cat(idx_batch)
        #idx_batch = idx_batch[dones_batch]
        
        action_batch = torch.stack(action_batch).transpose(1,0)
        action_batch = [action_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        action_batch = torch.cat(action_batch)
        #action_batch = action_batch[dones_batch]
        
        state_values_batch = torch.stack(state_values_batch).transpose(1,0)
        state_values_batch = [state_values_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        state_values_batch = torch.cat(state_values_batch)
        #state_values_batch = state_values_batch[dones_batch]

        
        
        last_idx_state_batch = []        
        for i in range(len(state_batch)): # for all timesteps/editing steps
            last_idxs = []
            for j in range(batch_size): # for all in the batch
                last_idxs.append(state_batch[i][j].shape[0])
            # works for [:last_idx_state_batch] or for [last_idx_state_batch - 1]
            last_idx_state_batch.append(last_idxs)
        
        # shape = (seq_length/number_edit steps, batch_size)
        last_idx_state_batch = torch.tensor(last_idx_state_batch).to(torch.long).to(config.device).transpose(1,0)
        last_idx_state_batch = [last_idx_state_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        last_idx_state_batch = torch.cat(last_idx_state_batch)
        #last_idx_state_last_time_step = last_idx_state_batch[-1,:]
        

        max_length = last_idx_state_batch.max()
        for i in range(len(state_batch)):  

            state_batch[i] = torch.nn.utils.rnn.pad_sequence(state_batch[i], batch_first=True).to(config.device)
            state_batch[i] = F.pad(state_batch[i], (0,0,0,max_length.item() - state_batch[i].shape[1],0,0))
            

        state_batch = torch.stack(state_batch).transpose(1,0)
        state_batch = [state_batch[i, :dones_batch_last_idx[i]] for i in range(batch_size)]
        state_batch = torch.cat(state_batch) 

        
        
        
        

        idx_array = torch.tensor(np.arange(idx_probs_on_policy.shape[0]))

        m_old_idx_on_policy = Categorical(idx_probs_on_policy)
        m_old_action_on_policy = Categorical(action_probs_on_policy)

        
        for k in range(self.K_epochs):

                if k == 0:
                    m_idx = Categorical(idx_probs_on_policy)
                    m_action = Categorical(action_probs_on_policy)
                else:
                    probs_unmasked, _, state_values_batch = self.calculate_action_values(state_batch)
                    idx_probs_on_policy, action_probs_on_policy = probs_unmasked

                    idx_probs_on_policy = idx_probs_on_policy[idx_array, last_idx_state_batch-1]
                    action_probs_on_policy = action_probs_on_policy[idx_array, last_idx_state_batch-1]
                    state_values_batch = state_values_batch[idx_array, last_idx_state_batch-1]
                    

                    idx_probs_exploration = self.mask_probs(idx_probs_on_policy, last_idx_state_batch-1)
                    action_probs_exploration = self.mask_probs(action_probs_on_policy, action_batch, last_idx_state_batch-1)

                    m_idx = Categorical(idx_probs_on_policy)
                    m_action = Categorical(action_probs_on_policy)
                    
                
                if self.action_mode == 'EDITING':
                    ratios = torch.exp(m_action.log_prob(action_batch) + m_idx.log_prob(idx_batch) - ( m_old_action_on_policy.log_prob(action_batch) + m_old_idx_on_policy.log_prob(idx_batch) ).detach())
                elif self.action_mode == 'APPEND':
                    ratios = torch.exp(m_action.log_prob(action_batch) - ( m_old_action_on_policy.log_prob(action_batch) ).detach())

                
                reward_batch = reward_batch.to(config.device)
                advantages = reward_batch - state_values_batch.detach().squeeze()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                if self.action_mode == 'EDITING':
                    self.loss = -torch.mean( torch.min(surr1.squeeze(), surr2.squeeze()) + self.entropy_weight*(m_action.entropy().squeeze() * m_idx.entropy().squeeze())) + 0.5*self.MseLoss(state_values_batch.squeeze(), reward_batch.squeeze())
                elif self.action_mode == 'APPEND':
                    self.loss = -torch.mean( torch.min(surr1.squeeze(), surr2.squeeze()) + self.entropy_weight*(m_action.entropy().squeeze())) + 0.5*self.MseLoss(state_values_batch.squeeze(), reward_batch.squeeze())
                    
                
                print('LOSS: ', self.loss)    
                super()._optimize()
                

        
        

