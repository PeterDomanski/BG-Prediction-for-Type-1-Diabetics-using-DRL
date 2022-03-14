# rl_time_series_forecasting

This project is about developing an RL-based framework for single or multi step ahead prediciton of time series data <br/>
Here is an overview of the proposed methodology:
![Alt text](./images/overview.png?raw=true "Methodology")

### Configuration options of the framework (conig.gin) <br/>
    - path_to_train_data [str]: specifiy path to csv training data set
    - path_to_eval_data [str]: specify path to csv testing data set 
        Note: if path is empty, training data is used for evaluation
    - setup [str]: specify setup (currently single_step and multi_step)
    - max_train_steps [int]: specify max number of training steps
    - eval_interval [int]: specify how often to evaluate
    - window_size [int]: specify length of window (single or multi step)
    - min_attribute_val [float]: specify minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: specify maximum value of attribute, e.g., CGM
    - forecasting_steps [int]: specify number of steps to forecast in multi step scenario
    - reward_def [str]: specify which reward function to use
        Options: abs_diff, linear, exponential
    - rl_algorithm [str]: specify which RL algorithm to use
        Options: ddpg, td3, reinforce, ppo, sac, dqn
        Note: dqn only for single step; tf_agents only supoort scalar actions 

### Visualization in Tensorboard
Navigate to logging directory of interest and type  (in the terminal)
```
tensorboard --logdir .
```
Or specify the absolute path to the directory, e.g., 
```
tensorboard --logdir /home/my_project/logs/log2022-03-11_11-49-13
```

### Program structure
![Alt text](./images/program_structure.png?raw=true "Program structure")
- dataset.py 
  - load data from csv file (time series data) <br/>
- environment.py
  - load data and set attributes, e.g., window size
  - define state := fixed-size window
  - reward := absolute difference of forecast and ground truth 
    - normalization of reward in range [0, 1]
    - idea
      - small absolute difference <-> high reward: r=1
      - large absolute difference <-> low reward: r=0
        - Absolute difference normalized in [0, 1] 
        - Linear definition  h
        ![Alt text](./images/reward_fct_lin.png?raw=true "Reward function exp")
        - Exponential definition
        ![Alt text](./images/reward_fct_exp.png?raw=true "Reward function exp")
    - Possible to try different values of `a` for slower/faster decrease or shift reward function in range [-1, 1] 
  - reset -> random starting point in (sequential) stream of the data
- agent.py
  - DDPG, A3C with RNN (LSTM/GRU)
- training.py
  - Reinforcement Learning loop
- evaluation.py
  - evaluation of forecasts
  - metrics: MAE, MSE, RMSE

### TODO's
- Continuous or discrete action space -> should prediction, here CGM, be continuous or discrete?
- More RL algorithms
- Reward function design, e.g., linear reward definition
- Hyperparameter tuning
- More configuration options

