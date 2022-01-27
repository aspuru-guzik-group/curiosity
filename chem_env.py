import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import QED
from rdkit import RDLogger
from rdkit.Chem import Draw

from selfies import encoder, decoder
from selfies_helper import is_finished
import helper
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from score_functions import calculate_pLogP
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import config
import pickle


def get_fingerprint(smile, radius, bits):
    m1 = Chem.MolFromSmiles(smile)
    if m1 is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=bits)
        x = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, x)
        return x
    else:
        return -1


# Celecoxib
fp1 = Chem.GetMorganFingerprint(Chem.MolFromSmiles('CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'), 2)


def calculate_score(smiles, mol, scoring_fnc='PLOGP'):
    if scoring_fnc == 'QED':
        # QED
        score = rdkit.Chem.QED.qed(mol)

    elif scoring_fnc == 'PLOGP':
        # PENALIZED LOG P
        score = calculate_pLogP(smiles)
        score = max([score, -10])
        score = score / 10

    elif scoring_fnc == 'SIMILARITY':
        # SIMILARITY
        fp2 = Chem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp1, fp2)

    return score


def update_molecule_render(img_array_m, title='Random sample', save_path=None, i_episode=-1):
    plt.figure(0)
    plt.clf()
    plt.title(title)
    plt.imshow(img_array_m)
    plt.pause(0.001)  # pause a bit so that plots are updated


class action_space(object):
    def __init__(self, actions):
        super(action_space)

        self.actions = actions
        self.n = len(self.actions)


class ChemEnv(object):
    def __init__(self, max_string_length, prediction_network=None, return_reward_total=True, start_symbol=True,
                 max_steps=75, scoring_fnc='PLOGP'):
        super(ChemEnv)

        self.max_string_length = max_string_length
        self.alphabet = helper.alphabet
        self.alphabet_size = helper.alphabet_size
        self.alphabet_dict = helper.alphabet_dict

        self.state_string = ''
        self.state = None
        self.action_space = action_space(self.alphabet)
        self.old_reward_total = 0

        self.img = None

        self.best_qed = -1000
        self.best_molecule_smiles = 'C'
        self.best_molecule_selfie = '[C]'

        self.return_reward_total = return_reward_total

        self.start_symbol = start_symbol

        self.n_steps = 0
        self.max_steps = max_steps

        # Disables WARNING: not removing hydrogen atom without neighbors
        RDLogger.DisableLog('rdApp.*')

        self.scoring_fnc = scoring_fnc

        self.normalizer = helper.Welford()

        self.done_intern = False

    def reset(self):
        self.n_steps = 0
        self.done = False
        self.done_intern = False


        self.state_string = '[C]'  # '[START PROXY]'

        if self.scoring_fnc == 'PLOGP':
            self.state_string = '[START PROXY]'  # '[S]'
        if self.scoring_fnc == 'SIMILARITY':
            self.state_string = encoder('CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F').split(']')[
                                    0] + ']'  # start with correct first symbol

        if self.state_string == '[START PROXY]':
            # start proxies one hot encoding is all 0
            self.state = np.zeros((1, len(helper.alphabet)))
            self.smiles_state_string = ''
            self.molecule = Chem.MolFromSmiles(self.smiles_state_string)
            self.old_reward_total = 0
        else:
            # first characer one hot encoding
            self.state = helper.selfie_string_to_one_hot(self.state_string)
            self.smiles_state_string = decoder(self.state_string)
            self.molecule = Chem.MolFromSmiles(self.smiles_state_string)

            self.old_reward_total = calculate_score(self.smiles_state_string, self.molecule,
                                                    scoring_fnc=self.scoring_fnc)
        return self.state

    def step(self, action):
        if self.done:
            if self.return_reward_total:
                return helper.selfie_string_to_one_hot(self.state_string), self.old_reward_total, self.done, None
            else:
                return helper.selfie_string_to_one_hot(self.state_string), 0, self.done, None

        ###################################################################################################
        ############################################ EXECUTE ACTION #######################################
        ###################################################################################################
        idx, symbol = action
        idx, symbol = idx.item(), symbol.item()
        action_symbol = self.action_space.actions[symbol]

        if len(self.state) == idx:
            # Append symbol to the end
            if action_symbol != '[STOP]':
                self.state_string += action_symbol
                self.state = helper.selfie_string_to_one_hot(self.state_string)
        else:
            # Substitute symbol
            if action_symbol != '[STOP]':
                self.state[idx] = helper.selfie_string_to_one_hot(action_symbol)
                self.state_string = helper.selfie_one_hot_to_selfie_string(self.state)

        molecule = Chem.MolFromSmiles(decoder(self.state_string))
        if molecule is not None:
            self.molecule = molecule
        ###################################################################################################
        ############################################ CALCULATE REWARD #####################################
        ###################################################################################################
        stop_condition = action_symbol == '[STOP]' or is_finished(self.state_string) or len(
            self.state) >= self.max_string_length or molecule is None

        if stop_condition:

            if action_symbol == '[STOP]':
                reward_total = self.old_reward_total
                reward = 0
            else:
                self.smiles_state_string = decoder(self.state_string)

                try:
                    reward, reward_total = self.calculate_reward(molecule)
                except:
                    reward, reward_total = 0, self.old_reward_total

            self.done = True
            self.info = None

        else:

            self.smiles_state_string = decoder(self.state_string)


            reward, reward_total = self.calculate_reward(molecule)

            self.done = False
            self.info = None

        if reward_total > self.best_qed:
            self.best_qed = reward_total
            self.best_molecule_smiles = self.smiles_state_string

        # counts how many edits were done
        self.n_steps += 1

        # action_symbol_one_hot = helper.selfie_string_to_one_hot(action_symbol)[0] #shape (1, 32) -> (32, ) this is how openai gym returns after step
        state_one_hot = helper.selfie_string_to_one_hot(self.state_string)

        if self.return_reward_total:
            return state_one_hot, reward_total, self.done, self.info
        else:
            return state_one_hot, reward, self.done, self.info

    def calculate_reward(self, molecule):
        try:
            if molecule is not None:

                reward_total = calculate_score(self.smiles_state_string, molecule, scoring_fnc=self.scoring_fnc)

            else:
                print('Failure case 1')
                print(self.smiles_state_string)
                print(self.molecule)
                print(self.state_string)
                print('##################### EXCEPTION #####################')
                error_mol = 'ERROR SELFIES: {} \n ERROR SMILES: {}'.format(self.state_string, self.smiles_state_string)
                with open('error.txt', "w") as text_file:
                    text_file.write(error_mol)
                reward = -self.old_reward_total - 1  # 0
                reward_total = -1  # self.old_reward_total
        except:
            print('Failure case 2')
            print(self.smiles_state_string)
            print(self.molecule)
            print(self.state_string)
            print(
                '##################### EXCEPTION ##########################################################################################################################')
            error_mol = 'ERROR SELFIES: {} \n ERROR SMILES: {}'.format(self.state_string, self.smiles_state_string)
            with open('error.txt', "w") as text_file:
                text_file.write(error_mol)
            reward_total = -1  # self.old_reward_total
            self.molecule = Chem.MolFromSmiles(self.smiles_state_string)

        reward = reward_total - self.old_reward_total
        self.old_reward_total = reward_total

        return reward, reward_total

    def render(self, mode=None, force_render=False, title='Random sample', save_path=None, i_episode=-1):
        if not self.done or force_render:
            self.img = Draw.MolToImage(self.molecule)

            img_array = np.array(self.img)
            update_molecule_render(img_array, title=title, save_path=save_path, i_episode=i_episode)
            return img_array
        else:
            return np.zeros((300, 300, 3))

    def save_env_state(self, experiment_path, id):
        best_result = 'SMILES STRING: {}  \n QED: {}'.format(self.best_molecule_smiles, self.best_qed)
        with open('{}/best_smiles_{}.txt'.format(experiment_path, id), "w") as text_file:
            text_file.write(best_result)

    def close(self):
        if self.img is not None:
            self.img.close()
            self.img = None

        self.reset()


######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################


class IntrinsicRewardChemParallel(object):
    def __init__(self, n_envs, prediction_network, max_string_length, max_train_buffer_len=1000, batch_size=32,
                 device='cpu', intrinsic_reward_weight=0, do_plot=False, start_symbol=True,
                 intrinsic_reward_type=None, scoring_fnc='PLOGP',  l2_curiosity=False, greedy_curiosity=True,
                 fingerprint_bits=16, fingerprint_radius=2, lsh_bits=16):
        super(IntrinsicRewardChemParallel)
        self.intrinsic_reward_weight = intrinsic_reward_weight

        self.n_envs = n_envs
        self.envs = [ChemEnv(max_string_length=max_string_length, return_reward_total=True, start_symbol=start_symbol,
                             scoring_fnc=scoring_fnc) for _ in range(n_envs)]
        self.prediction_network = prediction_network
        self.prediction_network_optim = optim.Adam(self.prediction_network.parameters(), lr=1e-3, weight_decay=0.005)

        self.old_reward_total = torch.tensor([env.old_reward_total for env in
                                              self.envs])  # torch.tensor(np.zeros((self.n_envs))) #torch.tensor([env.old_reward_total for env in self.envs])

        self.best_target = -2000

        self.batch_size = batch_size

        self.device = device

        self.do_plot = do_plot

        self.intrinsic_reward_type = intrinsic_reward_type
        self.previous_molecules_smiles = []
        self.previous_molecules_smiles_idx = 0

        self.scoring_fnc = scoring_fnc

        self.normalizer_reward = helper.Welford()
        self.normalizer_curiosity = helper.Welford()
        self.normalizer_pmdr = helper.Welford()

        self.l2_curiosity = l2_curiosity
        self.greedy_curiosity = greedy_curiosity

        self.best_sample = '[C]'

        self.c_error_train = []
        self.c_error_val = []
        self.pred_network_trained = False

        self.previously_encountered_mol_dict = {}

        self.state_count_dict = {}
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.lsh_rnd_matrix = np.random.randn(lsh_bits, fingerprint_bits)

    def lsh(self, array):
        return np.sign(np.dot(self.lsh_rnd_matrix, array))

    def save_smiles_for_distance_reward(self):
        max_length = 1000
        new_smiles_dict = {}
        for env in self.envs:
            mol = Chem.MolFromSmiles(env.smiles_state_string)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                new_smiles_dict[canonical_smiles] = Chem.GetMorganFingerprint(mol, 2)

        for smiles in new_smiles_dict.keys():
            self.previously_encountered_mol_dict[smiles] = new_smiles_dict[smiles]

            if smiles not in self.previous_molecules_smiles:
                if len(self.previous_molecules_smiles) < max_length:
                    self.previous_molecules_smiles.append(smiles)
                else:
                    self.previous_molecules_smiles_idx = (self.previous_molecules_smiles_idx + 1) % max_length
                    self.previous_molecules_smiles[self.previous_molecules_smiles_idx] = smiles

        self.previous_molecules_memory = {smiles: self.previously_encountered_mol_dict[smiles] for smiles in
                                          self.previous_molecules_smiles}

    def calculate_distance_to_previous_molecules(self, new_smiles_list):
        if len(self.previous_molecules_smiles) == 0:
            return torch.tensor(
                np.zeros((len(new_smiles_list))))  # [0 for i in range(len(self.previous_molecules_smiles))]

        all_distances = []
        for new_smiles in new_smiles_list:
            mol1 = Chem.MolFromSmiles(new_smiles)
            if mol1 is not None:
                fp1 = Chem.GetMorganFingerprint(mol1, 2)
                distances = []
                for old_smiles, fp2 in self.previous_molecules_memory.items():
                    # tanimoto_distance = np.abs(1/(TanimotoSimilarity(fp1, fp2) + 1e-1) -1)
                    # distances.append(tanimoto_distance)
                    distances.append((TanimotoSimilarity(fp1, fp2)))
                if len(distances) > 0:
                    # all_distances.append(np.min(distances))
                    all_distances.append(np.max(distances))
            else:
                # all_distances.append(100)
                all_distances.append(0)

        return torch.tensor(all_distances)

    def reset(self):
        self.loss = 0

        states = []
        self.old_reward_total = torch.tensor([env.old_reward_total for env in
                                              self.envs])  # torch.tensor(np.zeros((self.n_envs))) #torch.tensor([env.old_reward_total for env in self.envs])#torch.tensor(np.zeros((self.n_envs)))
        for env in self.envs:
            states.append(torch.tensor(env.reset()))
        states = np.stack(states)

        return torch.tensor(states).to(self.device), torch.tensor(
            [states.shape[1] - 1 for _ in range(states.shape[0])]).to(config.device)

    def step(self, actions, i_episode=0):

        states_one_hot = []
        dones = []
        infos = []
        targets = []
        counts = []

        ##################################################################################
        ################################# EXECUET ACTION #################################
        ##################################################################################
        # convert tuple of list to list of tuple
        idxs, actions = actions
        actions = [(idxs[i], actions[i]) for i in range(0, len(actions))]
        for i in range(self.n_envs):
            # state one hot is a list (hold by each environment) of the length of its selfie string
            state_one_hot, target, done, info = self.envs[i].step(actions[i])
            states_one_hot.append(state_one_hot)
            targets.append(torch.tensor(target).to(torch.float32))
            dones.append(done)
            infos.append(info)

            state_fingerprint = get_fingerprint(decoder(self.envs[i].state_string), self.fingerprint_radius,
                                                self.fingerprint_bits)
            if type(state_fingerprint) is not int:
                state_fingerprint = self.lsh(state_fingerprint).tostring()
            if state_fingerprint not in self.state_count_dict:
                self.state_count_dict[state_fingerprint] = 1
            else:
                if not done:
                    self.state_count_dict[state_fingerprint] += 1

            if target > self.best_target and target != 0:

                self.best_target = target
                self.best_sample = self.envs[i].state_string
                if self.do_plot:
                    self.envs[i].render(force_render=True, title='BEST SAMPLE: {}   {}'.format(
                        float(np.around(self.best_target, decimals=2)), self.envs[i].state_string))

        last_idx_list = torch.tensor([len(state_one_hot) - 1 for state_one_hot in states_one_hot]).to(config.device)
        targets = torch.stack(targets).to(config.device)
        states_one_hot = [torch.FloatTensor(state_one_hot) for state_one_hot in states_one_hot]
        ##################################################################################
        ################################# PREDICT VALUES #################################
        ##################################################################################
        if self.intrinsic_reward_type == 'COUNTING':
            for i in range(self.n_envs):
                state_fingerprint = get_fingerprint(decoder(self.envs[i].state_string), self.fingerprint_radius,
                                                    self.fingerprint_bits)
                if type(state_fingerprint) is not int:
                    state_fingerprint = self.lsh(state_fingerprint).tostring()
                counts.append(self.state_count_dict[state_fingerprint])

            intrinsic_reward = 1 / torch.sqrt(torch.tensor(counts).to(config.device, dtype=torch.float32))
        elif self.intrinsic_reward_type == 'MEMORY':
            intrinsic_reward = -self.calculate_distance_to_previous_molecules(
                [env.smiles_state_string for env in self.envs]).to(config.device)
            if np.all(np.array(dones)):
                self.save_smiles_for_distance_reward()
        elif self.intrinsic_reward_type == 'PREDICTION':
            last_idx_list = torch.tensor([len(state_one_hot) - 1 for state_one_hot in states_one_hot]).to(config.device)
            states_one_hot = [torch.FloatTensor(state_one_hot) for state_one_hot in states_one_hot]

            network_input = torch.nn.utils.rnn.pad_sequence(states_one_hot, batch_first=True)

            prediction = self.prediction_network(network_input.to(config.device))

            if self.l2_curiosity:
                loss = ((prediction[:, 0] - targets) ** 2)
            else:
                loss = torch.sqrt((prediction[:, 0] - targets) ** 2)

            mask_for_target_invalid = (targets != (torch.zeros_like(targets) - 1))
            loss = loss * mask_for_target_invalid

            if self.greedy_curiosity:
                loss_mask = targets >= targets.mean()
                loss = loss * loss_mask

            if self.pred_network_trained:
                intrinsic_reward = loss.detach()
            else:
                intrinsic_reward = torch.tensor(0.0)

        else:
            intrinsic_reward = torch.tensor([0]).to(self.device)

        ##################################################################################
        ################################# COMPOSE REWARD #################################
        ##################################################################################

        reward_total = targets

        self.old_reward_total = self.old_reward_total.to(config.device)
        rewards = reward_total - self.old_reward_total
        self.old_reward_total = reward_total

        rewards += self.intrinsic_reward_weight * intrinsic_reward

        self.prediction_network_optim.zero_grad()

        return states_one_hot, last_idx_list, rewards.to(
            torch.float32).detach().cpu().numpy(), dones, infos, targets, np.mean(
            intrinsic_reward.detach().cpu().numpy())

    def render(self, mode=None, force_render=False, n=1):
        for i, env in enumerate(self.envs):
            if i <= n:
                env.render(mode=mode, force_render=force_render)
            else:
                break

    def save_env_state(self, experiment_path, id):
        torch.save(self.prediction_network, '{}/pred_net_{}.pt'.format(experiment_path, id))
        for env in self.envs:
            env.save_env_state(experiment_path, id)

        with open('{}/state_count_dict_{}.pkl'.format(experiment_path, id), 'wb') as f:
            pickle.dump(self.state_count_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self):
        for env in self.envs:
            env.close()

    def load_model(self, path, id):
        self.prediction_network = torch.load('{}/pred_net_{}.pt'.format(path, id))
        self.prediction_network_optim = optim.Adam(self.prediction_network.parameters(), lr=1e-3)
        with open('{}/state_count_dict_{}.pkl'.format(path, id), 'rb') as f:
            self.state_count_dict = pickle.load(f)

        # self.selfies_dataset.load_data_from_file(path)
        # self.selfies_dataset_val.load_data_from_file(path, val=True)

    def push_to_train_buffer(self, x, y=None, path='.'):

        size = x.shape[0]
        train_size = int(0.8 * size)
        train_x, train_y = x[:train_size], y[:train_size]
        val_x, val_y = x[train_size:], y[train_size:]
        # self.selfies_dataset.push_data(train_x,train_y, path=path)
        # self.selfies_dataset_val.push_data(val_x,val_y, val=True, path=path)
