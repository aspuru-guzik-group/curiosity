# Curiosity in exploring the chemical space
This repository contains the code to reproduce experiments of "Curiosity in exploring the chemical space: Intrinsic rewards for molecular reinforcement learning"

## Setup
Install by cloning the repository and creating a environment using the requirements.txt
```
conda create --name intrinsic_rewad_mol_rl --file requirements.txt
conda activate intrinsic_rewad_mol_rl
```

## Example experiment
Run an experiment using 
```
python chem_ppo_parallel.py --intrinsic_reward_weight 0.1 --plot True --scoring_fnc PLOGP --discount_factor 1 --batch_size 64 --k_epochs 4 --intrinsic_reward_type COUNTING
```
which trains an RL agent to optimize the pLogP score using the count based intrinsic reward with a reward weight of 0.1
For the different reward types this should reproduce molecules similar to these ones
<p align="center">
  <img width="460" height="400" src="https://raw.githubusercontent.com/aspuru-guzik-group/curiosity/main/assets/curiosity_results.png?raw=true">
</p>
