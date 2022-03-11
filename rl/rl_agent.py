import tensorflow as tf
from tf_agents.agents.ddpg import actor_rnn_network, critic_rnn_network, ddpg_agent


def get_rl_agent(train_env):
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    time_step_spec = train_env.time_step_spec()

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

    agent.initialize()

    return agent
