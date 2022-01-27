Install by cloning the repository and creating a environment using the requirements.txt
```
conda create --name intrinsic_rewad_mol_rl --file requirements.txt
conda activate intrinsic_rewad_mol_rl
```
Run the experiments using 
```
python chem_ppo_parallel.py --intrinsic_reward_weight 0.1 --plot True --scoring_fnc PLOGP --discount_factor 1 --batch_size 64 --k_epochs 4 --intrinsic_reward_type COUNTING
```