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
    - preheat_phase [bool]: (off-policy algorithms) specify to use preheat phase or not
    - window_size [int]: specify length of window (single or multi step)
    - min_attribute_val [float]: specify minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: specify maximum value of attribute, e.g., CGM
    - forecasting_steps [int]: specify number of steps to forecast in multi step scenario
    - reward_def [str]: specify which reward function to use
        Options: abs_diff, linear, exponential
    - rl_algorithm [str]: specify which RL algorithm to use
        Options: ddpg, td3, reinforce, ppo, sac, dqn
        Note: dqn only for single step; tf_agents only supoort scalar actions 
        Note: On-policy: reinforce, ppo ; Off-policy: ddpg, td3, sac, dqn
    - max_window_count [int]: specify the maximum number of windows to use per training iteration
        Note: specify -1 if you want to use as much windows as possible with random starting point
    - env_implementation [str]: sepcify environment implementation to use
        Options: tf, gym
    - use_gpu [bool]: Specify to explictily use a GPU 

### Stand-alone scripts
1) metric_eval_csv.py
   - Script to calculate the standard metrics (MAE, MSE, RMSE, ...) for arbitrary data sequences 
   - Stand-alone script that can be called directly from the terminal, e.g. ```python metric_eval_csv.py```
   - It has the following configuration parameters (arguments of the python call)
     - csv_path [str]: path to csv data set
     - metrics [str]: metrics to evaluate, e.g., 'mse, rmse'
     - indices [str]: 'min_index, max_index' of samples to evaluate
     - strategy [str]: 'consecutive' or 'random' samples
     - setup [str]: 'single_step' or 'multi_step' scenario
   - Thus, a call of the script can look the following (including all arguments) <br/>
   ``` python metric_eval_csv.py --csv_path="." --metrics="mse,rmse" --indices="0,100" --strategy="consecutive" --setup="multi_step"```
2) uq_visualization.py
   - Script to visualize UQ (variance) samples of training iterations
   - Stand-alone script that can be called directly from the terminal, e.g. ```python uq_visualization.py```
   - It has the following configuration parameters (arguments of the python call)
     - csv_path [str]: path to csv data set
     - setup [str]: 'single_step' or 'multi_step' scenario
     - save_fig [str]: "True" or "False"
     - save_path [str]: path to store figure
   - Thus, a call of the script can look the following (including all arguments) <br/>
   ``` python uq_visualization.py --csv_path="." --setup="multi_step" --save_fig="True", --save_path="."```

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

### Testing
![Alt text](./images/unit_tests.png?raw=true "Unit tests")
We test the different algorithms in a single and multi step scenario on a debug data set (RasberryPi.csv). This 
data set can be found in the repository in ./data/RasberryPi.csv

### TODO's
- Use different replay buffers -> reverb replay buffer (deepmind)
- Curriculum learning
- Automatic Hyperparameter tuning
- Tests for single step and multi step scenarios with debug data and different RL algorithms

