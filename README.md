# rl_time_series_forecasting

This project is about developing an RL-based framework for single or multi step ahead prediciton of time series data <br/>
Here is an overview of the proposed methodology:
![Alt text](./overview.png?raw=true "Methodology")

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
