import argparse
import env_handler
from schoolclass import *
from env_handler import *
import chem_env as chem_env
from pathlib import Path
import config
import time
import neural_networks


################################## END IMPORTS #######################################
######################################################################################
################################## BEGINNING COMMAND LINE ARGUMENTS ##################

def main(job_id, intrinsic_reward_weight, do_plot, intrinsic_reward_type,
         scoring_fnc, max_string_length, discount_factor, k_epochs, num_episodes,
         l2_curiosity, greedy_curiosity, device, batch_size, load_previous_experiment,
         results_directory,
         n_epochs_pred_network, curiosity_buffer_size, entropy_weight, fingerprint_bits, fingerprint_radius, lsh_bits):
    job_id = int(job_id)
    results_directory = '{}/results/scoringFnc_{}/curiosityBufferSize_{}/maxSL_{}/intrinsicRewardType_{}/intrinsicRewardWeight_{}/discountFactor_{}/kEpochs_{}/l2Curosity_{}/greedyCuriosity_{}/nEpochsPredNetwork_{}/entropyWeight_{}/fingerprintBits_{}/fingerprintRadius_{}/lshBits_{}'.format(
        results_directory, scoring_fnc, curiosity_buffer_size, max_string_length,
        intrinsic_reward_type, intrinsic_reward_weight, discount_factor, k_epochs,
        l2_curiosity, greedy_curiosity, n_epochs_pred_network, entropy_weight,
        fingerprint_bits, fingerprint_radius, lsh_bits)
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    ################################## BEGINNING ENIVRONMENT ###################################
    #### BEGINNING NETWORK DEFINITION ####
    prediction_network = neural_networks.get_pred_network(device, batch_size).to(device)

    #### END NETWORK DEFINITION ####
    env = chem_env.IntrinsicRewardChemParallel(n_envs=batch_size, max_string_length=max_string_length,
                                               prediction_network=prediction_network,
                                               device=device,
                                               intrinsic_reward_weight=intrinsic_reward_weight, do_plot=do_plot, start_symbol=True,
                                               intrinsic_reward_type=intrinsic_reward_type,
                                               scoring_fnc=scoring_fnc, l2_curiosity=l2_curiosity,
                                               greedy_curiosity=greedy_curiosity,
                                               max_train_buffer_len=curiosity_buffer_size,
                                               fingerprint_bits=fingerprint_bits, fingerprint_radius=fingerprint_radius,
                                               lsh_bits=lsh_bits)
    n_actions = env.envs[0].action_space.n
    ################################## END ENIVRONMENT ###################################
    ######################################################################################
    ################################## BEGINNING AGENT ###################################

    #### BEGINNING NETWORK DEFINITION ####
    state_encoder = neural_networks.SELFIESEncoder(n_state_neurons=64, device=device, num_layers=1, dropout=0.1,
                                                   bidirectional=False).to(device)
    state_encoder_value_network = neural_networks.SELFIESEncoder(n_state_neurons=64, device=device, num_layers=2,
                                                                 dropout=0, bidirectional=False)
    critic = neural_networks.StateDecoder(state_encoder_value_network.n_state_neurons, 1, device=device)
    policy = neural_networks.StateDecoderTwoHeads(state_encoder.n_state_neurons, max_string_length, n_actions,
                                                  device=device)
    #### BEGINNING NETWORK DEFINITION ####
    networks = {'state_encoder': state_encoder, 'critic': critic, 'policy': policy,
                'state_encoder_value_network': state_encoder_value_network}
    lrs = {'state_encoder': 1e-3, 'critic': 1e-3, 'policy': 1e-3, 'state_encoder_value_network': 1e-3}

    ppo = PPOAgent(networks, device, environment='chem', identifier='PPO', lr=lrs, K_epochs=k_epochs,
                   discount_factor=discount_factor, entropy_weight=entropy_weight)

    #### BEGINNING LOAD OLD AGENT ####
    if load_previous_experiment and Path('{}/model_{}.pt'.format(results_directory, job_id)).is_file():
        ppo = torch.load('{}/model_{}.pt'.format(results_directory, job_id))
        env.load_model(results_directory, job_id)

        reward_history = np.load('{}/reward_hist_{}.npy'.format(results_directory, job_id)).tolist()
        molecule_property_history = np.load('{}/molecule_property_hist_{}.npy'.format(results_directory, job_id)).tolist()
        rings_list = np.load('{}/num_rings_{}.npy'.format(results_directory, job_id)).tolist()
        best_molecule_property_list = np.load('{}/best_molecule_property_{}.npy'.format(results_directory, job_id)).tolist()
        env.best_target = best_molecule_property_list[-1]
        best_molecule_property_list_final_molecule = np.load(
            '{}/best_molecule_property_final_molecule_{}.npy'.format(results_directory, job_id)).tolist()
        curiosity_error_hist = np.load('{}/curiosity_error_hist_{}.npy'.format(results_directory, job_id)).tolist()
        molecule_properties_list_raw = np.load('{}/score_raw_{}.npy'.format(results_directory, job_id), allow_pickle=True).tolist()
        rw_list_raw = np.load('{}/rewards_raw_{}.npy'.format(results_directory, job_id), allow_pickle=True).tolist()

        selfies_sample_list = np.load('{}/selfies_sample_list_{}.npy'.format(results_directory, job_id),
                                      allow_pickle=True).tolist()
        selfies_best_sample_list = np.load('{}/selfies_best_sample_list_{}.npy'.format(results_directory, job_id),
                                           allow_pickle=True).tolist()
        episode_start = np.load('{}/i_episode_{}.npy'.format(results_directory, job_id))[0]

        best_molecule_property_final_molecule = np.max(best_molecule_property_list_final_molecule)

    else:
        reward_history = []
        molecule_property_history = []
        rings_list = []
        best_molecule_property_list = []
        best_molecule_property_list_final_molecule = []
        best_molecule_property_final_molecule = -1000
        curiosity_error_hist = []
        molecule_properties_list_raw = []
        rw_list_raw = []
        selfies_sample_list = []
        selfies_best_sample_list = []
        episode_start = 0
    #### END LOAD OLD AGENT ####
    ################################## END AGENT #########################################
    ######################################################################################
    ################################## BEGINNING MAIN LOOP ###############################

    env.reset()
    base_reward = env.envs[0].old_reward_total

    if episode_start == num_episodes - 1:
        print('This run is already done!!!')
    else:
        for episode in range(episode_start, num_episodes):
            print('\nEpisode ', episode)
            time1 = time.time()

            state_batch, last_idx_list_all, idx_batch, action_batch, idx_probs_unmasked, idx_probs_masked, action_probs_unmasked, action_probs_masked, encoded_states_batch, state_values_batch, reward_batch, molecule_property_batch, dones_list, curiosity_error = env_handler.sample_env_trajectories_parallel(
                env, ppo, batch_size, device, i_episode=episode)

            state_batch_c = list(state_batch)
            for i in range(batch_size):
                selfies_sample_list.append(''.join(helper.one_hot_to_selfie(state_batch_c[-1][i].cpu().numpy())))
            print('Sample: ', selfies_sample_list[-1])
            selfies_best_sample_list.append(env.best_sample)

            #### BEGINNING CONVERT PADDED ARRAY TO LIST ####
            if intrinsic_reward_type == 'MEMORY':
                env.save_smiles_for_distance_reward()

            #### BEGINNING CALCULATE STATISTICS ####
            rw = np.array([rewards for rewards in reward_batch]).swapaxes(1, 0)


            total_average_reward = np.mean(rw.sum(-1) + base_reward)
            reward_history.append(total_average_reward)

            molecule_properties = np.array([molecule_property.detach().cpu().numpy() for molecule_property in molecule_property_batch]).swapaxes(1, 0)
            best_molecule_property = env.best_target

            if scoring_fnc == 'PLOGP':
                molecule_properties = molecule_properties * 10
                best_molecule_property = best_molecule_property * 10

            total_average_molecule_property = np.mean(molecule_properties[:, -1])
            molecule_property_history.append(total_average_molecule_property)

            best_molecule_property_list.append(best_molecule_property)

            best_molecule_property_final_molecule_ = np.max(molecule_properties[:, -1])
            if best_molecule_property_final_molecule_ > best_molecule_property_final_molecule:
                best_molecule_property_final_molecule = best_molecule_property_final_molecule_
            best_molecule_property_list_final_molecule.append(best_molecule_property_final_molecule)

            curiosity_error_hist.append(curiosity_error)
            #### END CALCULATE STATISTICS ####

            #### BEGINNING SHOW STATISTICS ####
            print('Molecule Property ', molecule_property_history[-1])
            print('BEST Molecule Property ', best_molecule_property_list[-1])
            if do_plot:
                helper.plot_live(molecule_property_history, 1, title=scoring_fnc)
                helper.plot_live(reward_history, 2, title="REWARD")
                helper.plot_live(curiosity_error_hist, 5, title="CURIOSITY")
                helper.plot_live(best_molecule_property_list, 4, title="BEST " + scoring_fnc)
                helper.plot_live(best_molecule_property_list_final_molecule, 6, title="BEST {} FINAL MOLECULE".format(scoring_fnc))

            print('Reward at episode ', episode, ': ', reward_history[-1])

            molecule_properties_list_raw.append(molecule_properties)
            rw_list_raw.append(rw)
            finalq = np.empty(len(molecule_properties_list_raw), dtype=object)
            finalq[:] = molecule_properties_list_raw
            finalr = np.empty(len(rw_list_raw), dtype=object)
            finalr[:] = rw_list_raw

            np.save('{}/reward_hist_{}.npy'.format(results_directory, job_id), reward_history)
            np.save('{}/molecule_property_hist_{}.npy'.format(results_directory, job_id), molecule_property_history)
            np.save('{}/num_rings_{}.npy'.format(results_directory, job_id), rings_list)
            np.save('{}/best_molecule_property_{}.npy'.format(results_directory, job_id), best_molecule_property_list)
            np.save('{}/best_molecule_property_final_molecule_{}.npy'.format(results_directory, job_id), best_molecule_property_list_final_molecule)
            np.save('{}/curiosity_error_hist_{}.npy'.format(results_directory, job_id), curiosity_error_hist)
            np.save('{}/score_raw_{}.npy'.format(results_directory, job_id), finalq)
            np.save('{}/rewards_raw_{}.npy'.format(results_directory, job_id), finalr)
            np.save('{}/i_episode_{}.npy'.format(results_directory, job_id), [episode])
            np.save('{}/selfies_sample_list_{}.npy'.format(results_directory, job_id), selfies_sample_list)
            np.save('{}/selfies_best_sample_list_{}.npy'.format(results_directory, job_id), selfies_best_sample_list)


            #### BEGINNING OPTIMIZE NETWORKS ####
            idx_probs_exploration, action_probs_exploration = idx_probs_masked, action_probs_masked
            idx_probs_on_policy, action_probs_on_policy = idx_probs_masked, action_probs_masked

            ppo.optimize(state_batch, last_idx_list_all, idx_batch, action_batch, idx_probs_on_policy,
                         idx_probs_exploration, action_probs_on_policy, action_probs_exploration, encoded_states_batch,
                         state_values_batch, reward_batch, dones_list)

            #### END OPTIMIZE NETWORKS ####

            #### BEGINNING SAVE EXPERIMENT ####
            ppo.snapshot_experiment(reward_history, results_directory, job_id)

            if ppo.save_directory_path is not None:
                env.save_env_state(ppo.save_directory_path, job_id)
            #### END SAVE EXPERIMENT ####

            time2 = time.time()
            print('TIME ', time2 - time1)

        print('\n')
        print('FINISHED!')


# python chem_ppo_parallel.py --intrinsic_reward_weight 0.1 --plot True --scoring_fnc PLOGP --discount_factor 1 --batch_size 64 --k_epochs 4 --intrinsic_reward_type COUNTING
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0)
    parser.add_argument('--intrinsic_reward_weight', default=0.1)
    parser.add_argument('--entropy_weight', default=0.01)
    parser.add_argument('--plot', default=False)
    parser.add_argument('--results_directory', default='results')
    parser.add_argument('--intrinsic_reward_type', default=None, choices=['COUNTING', 'MEMORY', 'PREDICTION'])
    parser.add_argument('--scoring_fnc', default='PLOGP', choices=['PLOGP', 'QED', 'SIMILARITY'])
    parser.add_argument('--max_string_length', default=35)
    parser.add_argument('--discount_factor', default=1)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--k_epochs', default=4)
    parser.add_argument('--l2_curiosity', default=True)
    parser.add_argument('--greedy_curiosity', default=False)
    parser.add_argument('--load_previous_experiment', default=True)
    parser.add_argument('--n_epochs_pred_network', default=0)
    parser.add_argument('--curiosity_buffer_size', default=1000)
    parser.add_argument('--fingerprint_bits', default=256)
    parser.add_argument('--fingerprint_radius', default=2)
    parser.add_argument('--lsh_bits', default=16)
    parser.add_argument('--num_episodes', default=3000)

    args = parser.parse_args()

    job_id = float(args.id)
    do_plot = (args.plot == 'True')
    intrinsic_reward_weight = float(args.intrinsic_reward_weight)
    entropy_weight = float(args.entropy_weight)
    results_directory = args.results_directory
    intrinsic_reward_type = args.intrinsic_reward_type
    scoring_fnc = args.scoring_fnc
    discount_factor = float(args.discount_factor)
    k_epochs = int(args.k_epochs)
    max_string_length = int(args.max_string_length)
    l2_curiosity = str(args.l2_curiosity) == 'True'
    greedy_curiosity = str(args.greedy_curiosity) == 'True'
    load_previous_experiment = str(args.load_previous_experiment) == 'True'
    n_epochs_pred_network = int(args.n_epochs_pred_network)
    curiosity_buffer_size = int(args.curiosity_buffer_size)
    fingerprint_bits = int(args.fingerprint_bits)
    fingerprint_radius = int(args.fingerprint_radius)
    lsh_bits = int(args.lsh_bits)
    batch_size = int(args.batch_size)
    num_episodes = int(args.num_episodes)
    device = config.device

    print('job_id: ', job_id)
    print('intrinsic_reward_type: ', intrinsic_reward_type)
    print('intrinsic_reward_weight: ', intrinsic_reward_weight)
    print('entropy_weight:', entropy_weight)
    print('do_plot: ', do_plot)
    print('results_directory: ', results_directory)
    print('scoring_fnc: ', scoring_fnc)
    print('max_string_length: ', max_string_length)
    print('discount_factor: ', discount_factor)
    print('k_epochs: ', k_epochs)
    print('l2_curiosity: ', l2_curiosity)
    print('greedy_curiosity: ', greedy_curiosity)
    print('load_previous_experiment: ', load_previous_experiment)
    print('n_epochs_pred_network:', n_epochs_pred_network)
    print('curiosity_buffer_size: ', curiosity_buffer_size)
    print('fingerprint_bits ', fingerprint_bits)
    print('fingerprint_radius ', fingerprint_radius)
    print('lsh_bits ', lsh_bits)
    print('batch_size', batch_size)
    print('num_episodes ', num_episodes)
    print('device ', device)

    ################################## END COMMAND LINE ARGUMENTS ########################
    ######################################################################################
    ################################## BEGINNING ENIVRONMENT #############################

    main(job_id=job_id, intrinsic_reward_weight=intrinsic_reward_weight, do_plot=do_plot,
         intrinsic_reward_type=intrinsic_reward_type,
         scoring_fnc=scoring_fnc, max_string_length=max_string_length, discount_factor=discount_factor,
         k_epochs=k_epochs, num_episodes=num_episodes, l2_curiosity=l2_curiosity, greedy_curiosity=greedy_curiosity,
         device=device, batch_size=batch_size, load_previous_experiment=load_previous_experiment, results_directory='.',
         n_epochs_pred_network=n_epochs_pred_network, curiosity_buffer_size=curiosity_buffer_size,
         entropy_weight=entropy_weight, fingerprint_bits=fingerprint_bits, fingerprint_radius=fingerprint_radius,
         lsh_bits=lsh_bits)
