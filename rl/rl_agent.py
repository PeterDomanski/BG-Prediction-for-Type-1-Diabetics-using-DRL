import tensorflow as tf
# import tf_agents
from absl import logging
from tf_agents.agents.ddpg import actor_rnn_network, critic_rnn_network, ddpg_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network, q_rnn_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils


def get_rl_agent(train_env, rl_algorithm="ddpg", use_gpu=False):
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    time_step_spec = train_env.time_step_spec()

    if rl_algorithm == "ddpg":
        with strategy.scope():
            actor_net = actor_rnn_network.ActorRnnNetwork(observation_spec,
                                                          action_spec,
                                                          input_fc_layer_params=(256, 256),
                                                          lstm_size=(64, ),
                                                          output_fc_layer_params=(256, 256),
                                                          activation_fn=tf.keras.activations.relu)
            critic_net = critic_rnn_network.CriticRnnNetwork((observation_spec, action_spec),
                                                             lstm_size=(64, ),
                                                             observation_fc_layer_params=(256, 256),
                                                             action_fc_layer_params=(128,),
                                                             joint_fc_layer_params=(256, 256),
                                                             output_fc_layer_params=(256, 256),
                                                             activation_fn=tf.keras.activations.relu,
                                                             kernel_initializer='glorot_uniform',
                                                             last_kernel_initializer='glorot_uniform'
                                                             )
            train_step = train_utils.create_train_step()

        agent = ddpg_agent.DdpgAgent(
            time_step_spec,
            action_spec,
            actor_net,
            critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(),
            critic_optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100,
            train_step_counter=train_step
        )
    elif rl_algorithm == "sac":
        # Hyperparameter setting as in https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial
        with strategy.scope():
            critic_net = critic_rnn_network.CriticRnnNetwork(
                (observation_spec, action_spec),
                lstm_size=(256,),
                observation_fc_layer_params=(256, 256),
                action_fc_layer_params=(128, ),
                joint_fc_layer_params=(256, 256),
                output_fc_layer_params=(256, 256),
                activation_fn=tf.keras.activations.relu,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform'
            )
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                observation_spec,
                action_spec,
                lstm_size=(256,),
                input_fc_layer_params=(256, 512, 256),
                output_fc_layer_params=(256, 512, 256),
                activation_fn=tf.keras.activations.relu,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
            )
            train_step = train_utils.create_train_step()

        agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            target_update_period=100,
            target_update_tau=0.005,
            gamma=1.0,
            reward_scale_factor=1.0,
            train_step_counter=train_step
        )

    elif rl_algorithm == "ppo":
        with strategy.scope():
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                observation_spec,
                action_spec,
                lstm_size=(64,),
                input_fc_layer_params=(256, 256),
                output_fc_layer_params=(256, 256),
                activation_fn=tf.keras.activations.relu,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
            )
            value_net = value_rnn_network.ValueRnnNetwork(
                observation_spec,
                input_fc_layer_params=(256, 256),
                lstm_size=(64,),
                output_fc_layer_params=(256, 256),
                activation_fn=tf.keras.activations.relu
            )
            train_step = train_utils.create_train_step()

        agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer=tf.keras.optimizers.Adam(),
            actor_net=actor_net,
            value_net=value_net,
            train_step_counter=train_step
        )
    elif rl_algorithm == "dqn":
        with strategy.scope():
            q_net = q_rnn_network.QRnnNetwork(
                observation_spec,
                action_spec,
                input_fc_layer_params=(256, 256),
                lstm_size=(64,),
                output_fc_layer_params=(256, 256),
                activation_fn=tf.keras.activations.relu,
            )
            train_step = train_utils.create_train_step()
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_net,
            optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100,
            train_step_counter=train_step
        )
    elif rl_algorithm == "td3":
        with strategy.scope():
            actor_net = actor_rnn_network.ActorRnnNetwork(observation_spec,
                                                          action_spec,
                                                          lstm_size=(64, ),
                                                          activation_fn=tf.keras.activations.relu)
            critic_net = critic_rnn_network.CriticRnnNetwork((observation_spec, action_spec),
                                                             lstm_size=(64,))
            train_step = train_utils.create_train_step()

        agent = td3_agent.Td3Agent(
            time_step_spec,
            action_spec,
            actor_net,
            critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(),
            critic_optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100,
            actor_update_period=100,
            train_step_counter=train_step
        )
    elif rl_algorithm == "reinforce":
        # Hyperparameter setting as in https://www.tensorflow.org/agents/tutorials/6_reinforce_tutorial
        with strategy.scope():
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                observation_spec,
                action_spec,
                lstm_size=(64,),
                input_fc_layer_params=(256, 256),
                output_fc_layer_params=(256, 256),
                activation_fn=tf.keras.activations.relu,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
            )
            value_net = value_rnn_network.ValueRnnNetwork(observation_spec,
                                                          input_fc_layer_params=(256, 256),
                                                          lstm_size=(64,),
                                                          output_fc_layer_params=(256, 256),
                                                          activation_fn=tf.keras.activations.relu,
                                                          )
            train_step = train_utils.create_train_step()

        agent = reinforce_agent.ReinforceAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            value_network=value_net,
            normalize_returns=True,
            train_step_counter=train_step
        )
    else:
        logging.info("Rl algorithm {} is not supported".format(rl_algorithm))

    agent.initialize()

    return agent
