import tensorflow as tf
import tf_agents
from absl import logging
from tf_agents.agents.ddpg import actor_rnn_network, critic_rnn_network, ddpg_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network, q_rnn_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils


def get_rl_agent(train_env, rl_algorithm="ddpg"):
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    time_step_spec = train_env.time_step_spec()

    if rl_algorithm == "ddpg":
        actor_net = actor_rnn_network.ActorRnnNetwork(observation_spec, action_spec)
        critic_net = critic_rnn_network.CriticRnnNetwork((observation_spec, action_spec),
                                                         lstm_size=(40, ))

        agent = ddpg_agent.DdpgAgent(
            time_step_spec,
            action_spec,
            actor_net,
            critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(),
            critic_optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100
        )
    elif rl_algorithm == "sac":
        critic_net = critic_rnn_network.CriticRnnNetwork(
            (observation_spec, action_spec),
            lstm_size=(40,)
        )
        actor_net = tf_agents.networks.actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            lstm_size=(40,),
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )

        agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(),
            critic_optimizer=tf.keras.optimizers.Adam(),
            alpha_optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100
        )

    elif rl_algorithm == "ppo":
        actor_net = tf_agents.networks.actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            lstm_size=(40,)
        )
        value_net = value_rnn_network.ValueRnnNetwork(
            observation_spec,
        )
        agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer=tf.keras.optimizers.Adam(),
            actor_net=actor_net,
            value_net=value_net
        )
    elif rl_algorithm == "dqn":
        q_net = q_rnn_network.QRnnNetwork(
            observation_spec,
            action_spec,
            lstm_size=(40,)
        )
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_net,
            optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100
        )
    elif rl_algorithm == "td3":
        actor_net = actor_rnn_network.ActorRnnNetwork(observation_spec, action_spec)
        critic_net = critic_rnn_network.CriticRnnNetwork((observation_spec, action_spec),
                                                         lstm_size=(40,))

        agent = td3_agent.Td3Agent(
            time_step_spec,
            action_spec,
            actor_net,
            critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(),
            critic_optimizer=tf.keras.optimizers.Adam(),
            target_update_period=100,
            actor_update_period=100,
        )
    elif rl_algorithm == "reinforce":
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            lstm_size=(40,))
        value_net = value_rnn_network.ValueRnnNetwork(observation_spec)
        agent = reinforce_agent.ReinforceAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            optimizer=tf.keras.optimizers.Adam(),
            value_network=value_net
        )
    else:
        logging.info("Rl algorithm {} is not supported".format(rl_algorithm))

    agent.initialize()

    return agent
