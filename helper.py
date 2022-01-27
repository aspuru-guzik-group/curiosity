import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
from selfies import encoder, decoder, selfies_alphabet


# from selfies import get_semantic_robust_alphabet as selfies_alphabet


def discount_rewards(rewards, gamma):
    running_add = 0
    for i in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[i]
        rewards[i] = running_add

    return rewards


class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at: 
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


def plot_live(rewards, window_id=5, width=None, title=''):
    if width is not None:
        rewards = rewards[-width:]
    plt.figure(window_id)
    plt.clf()
    rewards = torch.FloatTensor(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(rewards.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99) + rewards[0], means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_live_train_val(train_hist, val_hist, window_id=5, width=None, title=''):
    matplotlib.use("Qt4agg")
    plt.figure(window_id)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(train_hist)
    plt.plot(val_hist)
    # Take 100 episode averages and plot them too

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_live_scatter(rewards, rewards2, window_id=5, width=None, title=''):
    matplotlib.use("Qt4agg")

    plt.figure(window_id)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.scatter(rewards, rewards2)
    # Take 100 episode averages and plot them too

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_live_hist(rewards, i_episode, window_id=5, n_bins=10, title=''):
    matplotlib.use("Qt4agg")

    plt.figure(window_id)
    plt.clf()
    rewards = torch.FloatTensor(rewards)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.hist(rewards.numpy(), bins=n_bins)
    # if i_episode in [100, 200, 300, 400, 500, 700, 1200, 1700, 2000, 2500]:
    #    plt.savefig('./hist_with_e_{}_cweight_0.png'.format(i_episode))

    plt.pause(0.001)  # pause a bit so that plots are updated


##################################################### START CONVERSION HELPER #####################################################
alphabet = ['[#N]'] + selfies_alphabet() + ['[STOP]']
alphabet_size = len(alphabet)
alphabet_dict = {character: i for i, character in enumerate(alphabet)}


def character_to_one_hot(character):
    one_hot_encoding = np.zeros(alphabet_size)
    one_hot_encoding[alphabet_dict[character]] = 1
    return one_hot_encoding


def one_hot_to_character(one_hot_character):
    return alphabet[np.where(one_hot_character == 1)[0][0]]


def one_hot_to_selfie(one_hot_selfie):
    return [one_hot_to_character(one_hot_selfie[i]) for i in range(one_hot_selfie.shape[0])]


##### SMILE STRING TO ... #####

def smile_string_list_to_selfie_string_list(smile_string_list):
    return [encoder(smile_string) for smile_string in smile_string_list]


def smile_string_to_selfie_one_hot(smile_string):
    selfie_string = encoder(smile_string)
    return selfie_string_to_one_hot(selfie_string)


def smile_string_list_to_selfie_one_hot_list(smile_string_list, do_pad=False):
    selfie_one_hot_list = [torch.tensor(smile_string_to_selfie_one_hot(smile_string)) for smile_string in
                           smile_string_list]

    if do_pad:
        selfie_one_hot_list = torch.nn.utils.rnn.pad_sequence(selfie_one_hot_list, batch_first=True)

    return selfie_one_hot_list


#### SELFIE STRING TO ... ####

def selfie_string_to_one_hot(selfie_string):
    selfie_string_list_representation = ['[' + character for character in selfie_string.split('[')][1:]
    string_length = len(selfie_string_list_representation)

    one_hot_string = np.empty((string_length, alphabet_size))
    for i, character in enumerate(selfie_string_list_representation):
        one_hot_string[i] = character_to_one_hot(character)

    return one_hot_string


def selfie_string_list_to_one_hot(selfie_string_list):
    return [selfie_string_to_one_hot(selfie_string) for selfie_string in selfie_string_list]


def selfie_string_list_to_smile_string_list(selfie_string_list):
    return [decoder(selfie_string) for selfie_string in selfie_string_list]


#### SELFIE ONE HOT TO ... ####

def selfie_one_hot_to_selfie_string(selfie_one_hot):
    return ''.join(one_hot_to_selfie(selfie_one_hot))


def selfie_one_hot_list_to_selfie_string_list(selfie_one_hot_list):
    return [selfie_one_hot_to_selfie_string(selfie_one_hot) for selfie_one_hot in selfie_one_hot_list]


def selfie_one_hot_to_smiles_string(selfie_one_hot):
    selfie_string = selfie_one_hot_to_selfie_string(selfie_one_hot)
    return decoder(selfie_string)


def selfie_one_hot_list_to_smile_string_list(selfie_one_hot_list):
    return [selfie_one_hot_to_smiles_string(selfie_one_hot) for selfie_one_hot in selfie_one_hot_list]


##################################################### END CONVERSION HELPER #####################################################


def reset_all_recurrent_modules(agent, batch_size=1):
    for key in agent.networks:
        network = agent.networks[key]
        if hasattr(network, 'is_recurrent'):
            network.reset_hidden_state(batch_size=batch_size)


def truncate_all_hidden_states(agent):
    for key in agent.networks:
        network = agent.networks[key]
        if hasattr(network, 'is_recurrent'):
            network.truncate_hidden_state()


def set_all_use_cell_states(agent, use_cell):
    for key in agent.networks:
        network = agent.networks[key]
        if hasattr(network, 'is_recurrent'):
            network.set_use_cell(use_cell)


def sync_all_cells(agent):
    for key in agent.networks:
        network = agent.networks[key]
        if hasattr(network, 'is_recurrent'):
            network.sync_cell()


def is_sequence_like(agent):
    for key in agent.networks:
        network = agent.networks[key]
        if hasattr(network, 'is_recurrent'):
            return True
    return False


def convert_lstm_to_lstm_cells(lstm):
    lstm_cells = nn.ModuleList([nn.LSTMCell(lstm.input_size, lstm.hidden_size)] +
                               ([nn.LSTMCell(lstm.hidden_size, lstm.hidden_size)] * (lstm.num_layers - 1)))

    key_names = lstm_cells[0].state_dict().keys()
    source = lstm.state_dict()
    for i in range(lstm.num_layers):
        new_dict = OrderedDict([(k, source["%s_l%d" % (k, i)]) for k in key_names])
        lstm_cells[i].load_state_dict(new_dict)

    return lstm_cells


def convert_lstm_cells_to_lstm(lstm_cells):
    lstm = nn.LSTM(lstm_cells[0].input_size, lstm_cells[0].hidden_size, len(lstm_cells))

    key_names = lstm_cells[0].state_dict().keys()
    lstm_dict = OrderedDict()
    for i, lstm_cell in enumerate(lstm_cells):
        source = lstm_cell.state_dict()
        new_dict = OrderedDict([("%s_l%d" % (k, i), source[k]) for k in key_names])
        lstm_dict = OrderedDict(list(lstm_dict.items()) + list(new_dict.items()))
    lstm.load_state_dict(lstm_dict)

    return lstm
