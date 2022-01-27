
from chem_ppo_parallel import main
import argparse, sys
import itertools

extrinsic_rewards=[True]
curiosity_weights=[0, 0.005, 0.01, 0.02, 0.04] 
use_previous_molecule_distance_rewards=[False]
scoring_fncs=['SIMILARITY']
action_modes=['APPEND']
discount_factors=[1.0] 
k_epochss=[4] 
env_agent_share_encoders=[False]
l2_curiositys=[False] 
winner_curiositys=[False] 
every_time_step_feedbacks=[True]
job_ids=[2,3]
curiosity_buffer_sizes=[20000]
entropy_weights=[0.01, 0.02, 0.03, 0.04, 0.05]
fingerprint_radiuses = [2]
fingerprint_bitss = [256]
lsh_bitss = [16, 32, 64, 128]


hyperparameter_dict_list = []


for hyperparameter in list(itertools.product(extrinsic_rewards, curiosity_weights, 
                            use_previous_molecule_distance_rewards, scoring_fncs, action_modes, 
                            discount_factors, k_epochss, env_agent_share_encoders, 
                            l2_curiositys, winner_curiositys, every_time_step_feedbacks, curiosity_buffer_sizes, job_ids, entropy_weights, fingerprint_radiuses, fingerprint_bitss, lsh_bitss)):
    extrinsic_reward, curiosity_weight, use_previous_molecule_distance_reward, scoring_fnc, action_mode, discount_factor, k_epochs, env_agent_share_encoder, l2_curiosity, winner_curiosity, every_time_step_feedback, curiosity_buffer_size, job_id, entropy_weight, fingerprint_radius, fingerprint_bits, lsh_bits = hyperparameter

    if curiosity_weight == 0 and lsh_bits > 16 or curiosity_weight > 0 and entropy_weight not in [0.01, 0.02]:
        continue

    hyperparameter_dict = {'extrinsic_reward':extrinsic_reward, 'intrinsic_reward_weight':curiosity_weight,'use_previous_molecule_distance_reward':use_previous_molecule_distance_reward,
                            'scoring_fnc':scoring_fnc, 'action_mode':action_mode, 'discount_factor':discount_factor, 'k_epochs':k_epochs, 'env_agent_share_encoder':env_agent_share_encoder,
                            'l2_curiosity':l2_curiosity, 'winner_curiosity':winner_curiosity, 'every_time_step_feedback':every_time_step_feedback, 'job_id':job_id, 
                            'curiosity_buffer_size':curiosity_buffer_size, 'entropy_weight':entropy_weight, 'fingerprint_radius':fingerprint_radius, 'fingerprint_bits':fingerprint_bits,
                            'lsh_bits': lsh_bits}

    hyperparameter_dict_list.append(hyperparameter_dict)



for i, hyperparameter_dict in enumerate(hyperparameter_dict_list):
    hyperparameter_dict['i']=i
    hyperparameter_dict_list[i] = hyperparameter_dict


print('LENGTH: ', len(hyperparameter_dict_list))
'''
for i in range(len(hyperparameter_dict_list)):
    print(hyperparameter_dict_list[i])
'''

def launch_main(i, results_directory):
    i = int(i)
    if i == len(hyperparameter_dict_list):
        i = 0

    job_id = hyperparameter_dict_list[i]['job_id']
    curiosity_weight = hyperparameter_dict_list[i]['intrinsic_reward_weight']
    entropy_weight= hyperparameter_dict_list[i]['entropy_weight']
    do_plot = False
    pca = 0
    extrinsic_reward = hyperparameter_dict_list[i]['extrinsic_reward']
    use_previous_molecule_distance_reward = hyperparameter_dict_list[i]['use_previous_molecule_distance_reward']
    scoring_fnc = hyperparameter_dict_list[i]['scoring_fnc']
    if scoring_fnc == 'SIMILARITY':
        max_string_length = 60
    else:
        max_string_length = 35
    action_mode = hyperparameter_dict_list[i]['action_mode']
    discount_factor = hyperparameter_dict_list[i]['discount_factor']
    k_epochs = hyperparameter_dict_list[i]['k_epochs']
    env_agent_share_encoder = hyperparameter_dict_list[i]['env_agent_share_encoder']
    num_episodes = 1000
    l2_curiosity = hyperparameter_dict_list[i]['l2_curiosity']
    winner_curiosity = hyperparameter_dict_list[i]['winner_curiosity']
    every_time_step_feedback = hyperparameter_dict_list[i]['every_time_step_feedback']
    curiosity_buffer_size = hyperparameter_dict_list[i]['curiosity_buffer_size']
    fingerprint_bits = hyperparameter_dict_list[i]['fingerprint_bits']
    fingerprint_radius = hyperparameter_dict_list[i]['fingerprint_radius']
    lsh_bits = hyperparameter_dict_list[i]['lsh_bits']


    print('job_id ', job_id, '\n',
        'intrinsic_reward_weight ', curiosity_weight,  '\n',
        'do_plot ', do_plot,  '\n',
        'pca ', pca,  '\n',
        'extrinsic_reward ', extrinsic_reward,  '\n',
        'use_previous_molecule_distance_reward ', use_previous_molecule_distance_reward,  '\n',
        'scoring_fnc ', scoring_fnc,  '\n',
        'max_string_length ', max_string_length,  '\n',
        'action_mode ', action_mode,  '\n',
        'discount_factor ', discount_factor,  '\n', 
        'k_epochs ', k_epochs,  '\n',
        'env_agent_share_encoder ', env_agent_share_encoder,  '\n',
        'num_episodes ', num_episodes,  '\n',
        'l2_curiosity ', l2_curiosity,  '\n',
        'winner_curiosity ', winner_curiosity, '\n',
        'every_time_step_feedback ', every_time_step_feedback, '\n',
        'curiosity_buffer_size ', curiosity_buffer_size, '\n',
        'fingerprint_bits ', fingerprint_bits,  '\n',
        'fingerprint_radius ', fingerprint_radius, '\n',
        'entropy_weight ', entropy_weight, '\n',
        'lsh_bits ', lsh_bits)
    print(hyperparameter_dict_list[i])


    n_epochs_pred_network = 10000

    main(job_id, curiosity_weight, do_plot, pca, extrinsic_reward, use_previous_molecule_distance_reward,
        scoring_fnc, max_string_length, action_mode, discount_factor, k_epochs, env_agent_share_encoder, num_episodes,
        l2_curiosity, winner_curiosity, device='cuda:0', batch_size=64, every_time_step_feedback = every_time_step_feedback, load_previous_experiment=True, 
        results_directory=results_directory, n_epochs_pred_network=n_epochs_pred_network, curiosity_buffer_size=curiosity_buffer_size, entropy_weight=entropy_weight, 
        fingerprint_radius = fingerprint_radius, fingerprint_bits=fingerprint_bits, lsh_bits = lsh_bits)





if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--id', default=0)
    parser.add_argument('--results_directory', default='.')
    args=parser.parse_args()
    job_id = int(args.id)
    results_directory = args.results_directory

    print('JOB ID: ', job_id)
    print('RESULTS DIRECTORY: ', results_directory)
    launch_main(job_id, results_directory)
    
    '''
    for i in range(1,len(hyperparameter_dict_list)):
        launch_main(i, '.')
    '''


#sudo apt install openbabel