# PYFLAGS
#Installation

Look up anaconda documentation to see how to use the environment.yml file to create a virtual environment. You may need to install a few more libraries after creating the environment using the pip command. In case the conda environment creation process fails to execute perfectly, the environment may have still been created and you can activate it and install whatever else is necessary.

## INTRO

This program is used to train two adversarial RL agents against each other using self play. The training is broken down into alternating phases, where, after every 2500000 steps the training shifts between the blue agent and the red agent. 

## Getting started

`python3 guided_train.py`

This command starts the simulation loading the default pre-trained models trained against the programmed AI - "evade" and "fire models. It trains it for 500 episodes by default.

`python3 guided_train.py --blue-model=<model name> --red-model=<model name> --n-timesteps=<no. steps>`

This command can be used to specify which models should be loaded, and for how many timesteps should the training take place.

## Rewards

Bring flag back to base: +100
Opponent brings flag to its base: -100

###############################################

## CTF.py

- The key componenets of the environment file are action_space, observation_space, reset(), step(), reward()

- reset(): Reads in all the relevant data into appropriate arrays such as allTanks, allOstacles, allBases, allFlags, etc.

- step(): recieves an action in range (-1, 1). For Tank speed and Firing actions these are changed to the range (0,1). The rest of the program mostly checks for object collisions and gets the reward.

- Since we are on a 1-on-1 setting, only local_rewards are changed, not global.

- If any approach requires a shaping reward, that can be added at line 339.

- In case changes need to be made to the reward, it can be found in lines 457-483 and 520-527.

- The control functions can be used to get the action that the rules-based expert would have taken for the Blue Tank. Some randomness can be added to it by changing the value of the gen_eps variable.

##################################################

## guided_train.py

- The command line parameters, such as which attacking and evading model to use and for how many episodes, is parsed

- The callback function is used to save the model every few episodes. 

- Each of the two models are loaded. The hyperparameters set are fairly well tuned. PPO2 model is used for training.

- Two Threads are created to run each model. The models communicate with the guided_train program through a message Queue where they send the actions to be taken in the environment and the guided_train program sends those actions to the environment to be executed and communicates back to each model, the reward it received and its new observation after taking the action.

- This is how the self-play is able to take place.

####################################################

## self_play_ppo2.py

- The learn() and _run() functions are the main crux of this program.

- Learn() calls the runner to execute run() for n_steps. It collects all that data and uses it to update the network and then repeats the process again for total_timesteps iterations.

- the allow_update variable is used to decide which model is trained in the current cycle and which one is only used for prediction.

- The model provides an action to take and gives the value of a state based on the provided observation. This is communicated to the guided_train program where the actions from all models are combined into one and executed in the environment. Each model is then told the reward it recieved and its respective new observations. 

- If at all you wish to learn against the rule-based AI instead of the red tank model, the control functions in CTF.py can be used to get the expert action and you can rewrite the action predicted by model_2 with this expert_action. 

-To try different variants of Reward phasing, modify the phase_condition() and get_phase_step() functions in the Runner class to specify the condition in which further phasing should take place towards the true reward function and by how much should the reward phasing progress by, respectively.

#####################################################

## rewards_1.txt

- saves the mean reward for model_1 per episode

#####################################################

## loss_1.txt

- saves the mean reward for model_1 per episode

####################################################

## Changes to policy and architecture

- If changes to the network architecture or any policies are to be made, most of the programs relating to it can be found in the commons folder in the stable_baselines folder.

- To make the changes, make a local copy of that file into your project folder, or make a subclass of that class in your project folder, import this file into your programs and make required changes to these files. This way the original code does not need to be tampered with.

