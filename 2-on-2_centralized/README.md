# PYFLAGS
## INTRO

This program is used to train an Evading agent against an Attacking agent using self play. The training is broken down into alternating phases, where, the timesteps in the range 0%-25% and 50%-75% of the total timesteps is used to train the evasion agent and the timesteps in the range 25%-50% and 75%-100% of the total timesteps is used to train the attack/firing agent. 

## Getting started

`python3 guided_train.py`

This command starts the simulation loading the default pre-trained models trained against the programmed AI - "evade" and "fire models. It trains it for 500 episodes by default.

`python3 guided_train.py --evade-model=<model name> --fire-model=<model name> --n-timesteps=<no. steps>`

This command can be used to specify which evasion model or firing model should be loaded, and for how many timesteps should the training take place.

## Rewards
Fire agent:

-Fire a bullet: -1
-Shoot a tank: +20

Evade agent:

-Fire a bullet: -1
-Shot by a tank: -20

Combined agent:
-Fire a bullet: -1
-Shoot a tank: +20
-Shot by a tank: -20


###############################################

## CTF.py

- The key componenets of the environment file are action_space, observation_space, reset(), step(), reward()

- The models to be merged are "attack" and "evade". These can be loaded into the environment file if required by uncommenting lines 109-115.

- reset(): Reads in all the relevant data into appropriate arrays such as allTanks, allOstacles, allBases, allFlags, etc.

- step(): recieves an action in range (-1, 1). For Tank speed and Firing actions these are changed to the range (0,1). The rest of the program mostly checks for object collisions and gets the reward.

- Since we are on a 1-on-1 setting, only local_rewards are changed, not global.

- The predictions of the learnt attack and evade models can be obtained by the commands mentioned in lines 231-232, or 323-324.

- If any approach requires a shaping reward, that can be added at line 329.

- In case changes need to be made to the reward, it can be found in lines 263, 451-469.

- I am pretty sure the rest of the program can be left mostly as is. The control functions can be used to get the action that the rules-based expert would have taken for the Blue Tank. Some randomness can be added to it by changing the value of the gen_eps variable.

##################################################

## guided_train.py

- The command line parameters, such as which attacking and evading model to use and for how many episodes, is parsed

- The callback function is used to save the model every few episodes. 

- Each of the two models are loaded. The hyperparameters set are fairly well tuned. PPO2 model is used for training.

- Two Threads are created to run each model. The models communicate with the guided_train program through a message Queue where they send the actions to be taken in the environment and the guided_train program sends those actions to the environment to be executed and communicates back to each model, the reward it received and its new observation after taking the action.

- This is how the self-play is able to take place.

####################################################

## customPPO2.py

- The learn() and _run() functions are the main crux of this program.

- Learn() calls the runner to execute run() for n_steps. It collects all that data and uses it to update the network and then repeats the process again for total_timesteps iterations.

- the allow_update variable is used to decide which model is trained in the current cycle and which one is only used for prediction. Since we are only trying to guide the learning of model_1 using experts and model_2 is set to the fire_model, we do not need to further train model_2. Hence, allow_update is set to 1 only for model_1.

- The model provides an action to take and gives the value of a state based on the provided observation. This is communicated to the guided_train program where the actions from all models are combined into one and executed in the environment. Each model is then told the reward it recieved and its respective new observations. 

- If at all you wish to learn against the rule-based AI instead of the fire model, the control functions in CTF.py can be used to get the expert action and you can rewrite the action predicted by model_2 with this expert_action. 

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

