# Blood Glucose Prediction for Type-1 Diabetics using Deep Reinforcement Learning

[![Linux](https://svgshare.com/i/Zhy.svg)](https://ubuntu.com/download)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)


Deep Reinforcement Learning-based framework for single- or multi-step ahead predictions of time series data, 
e.g. (continuous) blood glucose values of type 1 diabetics. <br/>
Here is an overview of the proposed methodology:

![Alt text](./images/overview.png?raw=true "Methodology")

### Ohio T1DM dataset
Learn more about the Ohio T1DM dataset for blood glucose prediction [here](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html). <br/>

### Dependencies
    - tensorflow 2.8.0
    - tensorboard 2.8.0
    - tf-gents 0.12.0
    - gym 0.23.0
    - pandas 1.1.5
    - numpy 1.21.5
    - matplotlib 3.5.1
    - absl-py 1.0.0
    - gin-config 0.5.0
Install dependencies using the following command:
```pip install -r requirements.txt```

### Configuration options of the framework (see conig.gin) <br/>
#### General settings
    - path_to_train_data [str]: specifiy path to csv training data set
    - path_to_eval_data [str]: specify path to csv testing data set 
        Note: if path is empty, training data is used for evaluation
    - normalization [str]: specify normalization method  (options: min_max, z_score, none)
    - setup [str]: specify setup (currently single_step and multi_step)
    - rl_algorithm [str]: specify RL algorithm to use   (options: ddpg, td3, reinforce, ppo, sac, dqn)
        Note: dqn only for single step; tf_agents only supoort scalar actions 
        Note: On-policy: reinforce, ppo ; Off-policy: ddpg, td3, sac, dqn
    - env_implementation [str]: specify environment implementation to use   (options: tf, gym)
    - use_gpu [bool]: Specify to use GPU(s) 
    - multi_task [bool]: If True multi-task setup (training on multiple patients simultaneously)
#### DRL training settings
    - max_train_steps [int]: specify max number of training steps
    - eval_interval [int]: specify how often to evaluate
    - pretraining_phase [bool]: (off-policy algorithms) specify to use pretraining phase or not
    - restore_dir [str]: path to directory with weights and biases to restore (no restoring if empty str) 
    - layers_to_train [str]: if multi-task with resotring this str specifies which layers to train
       (options: last, dec_last, lstm_dec_last)
#### Single step settings
    Gym environment settings 
    ------------------------------
    - window_size [int]: input window size
    - min_attribute_val [float]: minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: maximum value of attribute, e.g., CGM
    - reward_def [str]: specify reward function to use    (options: abs_diff, linear, exponential)
    - max_window_count [int]: specify the maximum number of windows to use per training iteration
        Note: specify -1 if you want to use as much windows as possible with random starting point


     TF environment settings
    ------------------------------
    - window_size [int]: input window size
    - min_attribute_val [float]: minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: maximum value of attribute, e.g., CGM
    - max_window_count [int]: specify the maximum number of windows to use per training iteration
        Note: specify -1 if you want to use as much windows as possible with random starting point
    - batch_size [int]: specify batch size
    - state_type [str]: specify state type (options: skipping, no_skipping, single_step_shift)
#### Multi step settings
#### Gym environment settings
    - window_size [int]: input window size
    - forecasting_steps [int]: number of steps to forecast
    - min_attribute_val [float]: minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: maximum value of attribute, e.g., CGM
    - reward_def [str]: Specify reward function to use    (options: abs_diff, linear, exponential)
    - max_window_count [int]: specify the maximum number of windows to use per training iteration
        Note: specify -1 if you want to use as much windows as possible with random starting point
#### TF environment settings
    - window_size [int]: input window size
    - pred_horizon [int]: number of steps to forecast
    - min_attribute_val [float]: minimum value of attribute, e.g., CGM
    - max_attribute_val [float]: maximum value of attribute, e.g., CGM    
    - max_window_count [int]: specify the maximum number of windows to use per training iteration
        Note: specify -1 if you want to use as much windows as possible with random starting point
    - batch_size [int]: specify batch size
    - state_type [str]: specify state type (options: skipping, no_skipping, single_step_shift)

### Run in terminal
Set configuration options in config.gin and run the following command in the terminal:
```python main.py```
### Run in Google colab
- main.py: change path to config.gin (absolute path, e.g., '/content/rl_time_series_forecasting/config.gin')
- config.gin: change path to datasets (absolute path, e.g., '/content/rl_time_series_forecasting/data/540-ws-training.csv')
- Jupyter notebook code
  - ``` !pip install -r /content/BG-Prediction-for-Type-1-Diabetics-using-DRL/requirements.txt```
  - ``` !python /content/BG-Prediction-for-Type-1-Diabetics-using-DRL/main.py```

### Visualization in Tensorboard
Navigate to logging directory of interest and type  (in the terminal)
```
tensorboard --logdir .
```
Or specify the absolute path to the directory, e.g., 
```
tensorboard --logdir /home/my_project/logs/my_log_dir
```

### Stand-alone scripts (in evaluation directory)
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
   ``` python metric_eval_csv.py --csv_path="./data.csv" --metrics="mse,rmse" --indices="0,100" --strategy="consecutive" --setup="multi_step"```
   
   
2) uq_visualization.py
   - Script to visualize UQ (variance) samples of training iterations
   - Stand-alone script that can be called directly from the terminal, e.g. ```python uq_visualization.py```
   - It has the following configuration parameters (arguments of the python call)
     - csv_path [str]: path to csv data set
     - setup [str]: 'single_step' or 'multi_step' scenario
     - vis_type [str]: different types of plots ('vis_eval_samples', 'vis_avg_training', 'vis_avg_forecasting')
     - vis_steps [int]: number of steps to visualize (multiple of eval freq, e.g. vis_steps * 500)
     - vis_std [bool]: if True visualize std otherwise visualize variance
     - vis_forecasting_error [bool]: if True additionally visualize forecasting error
     - error_metric [str]: metric to visualize if vis_forecasting_error is True
     - y_lim [int]: limit of y-axis for vis_avg_forecasting
     - dataset_path [str]: path to (training) dataset to visualize windows with min. / max. error
     - save_fig [str]: "True" or "False"
     - save_path [str]: path to store figure
   - Thus, a call of the script can look the following (including all arguments) <br/>
   ``` python uq_visualization.py --csv_path="./data.csv" --setup="multi_step" --save_fig="True", --save_path="."```
   

3) seg_visualization.py
   - Script to calculate and visualize the Surveillance Error Grid (SEG)
   - Stand-alone script that can be called directly from the terminal, e.g. ```python seg_visualization.py```
   - It has the following configuration parameters (arguments of the python call)
     - csv_path [str]: path to csv data set
     - img_path [str]: path to seg image (superimposed with prediction results)
     - ph [int]: prediction horizon, e.g., 30 min
     - patient [int]: patient id
     - save_path [str]: path to store figure
   - Thus, a call of the script can look the following (including all arguments) <br/>
   ``` python uq_visualization.py --csv_path="./data.csv" --img_path="./seg.png" --ph=30, --patienth=570 --save_path="."```
   - Example:
   ![Alt text](./images/seg_30min_570.png?raw=true "SEG 30 min subject 570")




