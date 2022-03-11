# rl_time_series_forecasting

This project is about developing an RL-based framework for single or multi step ahead prediciton of time series data <br/>
Here is an overview of the proposed methodology:
![Alt text](./images/overview.png?raw=true "Methodology")

### Configuration options of the framework (conig.gin) <br/>
    - path_to_data [str]: specifiy path to csv data set
    - setup [str]: specify setup (currently single_step and multi_step)

### Visualization in Tensorboard
Navigate to logging directory of interest and type  (in the terminal)
```
tensorboard --logdir .
```
Or specify the total path to the directory, e.g., 
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
      - small absolute difference <-> high reward
      - large absolute difference <-> low reward
      ![Alt text](./images/reward_fct.png?raw=true "Reward function")
    - Possible to try different values of `a` for slower/faster decrease or shift reward function in range [-1, 1] 
  - reset -> random starting point in (sequential) stream of the data
- agent.py
  - DDPG, A3C with RNN (LSTM/GRU)
- training.py
  - Reinforcement Learning loop
- evaluation.py
  - evaluation of forecasts
  - metrics: MAE, MSE, RMSE

