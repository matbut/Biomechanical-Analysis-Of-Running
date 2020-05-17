"""
solving pendulum using actor-critic model
"""

import gym

import random

import numpy as np
import pybullet_envs  # do not delete, necessary to make gym env

from actor import Actor
from buffer import Buffer
from critic import Critic


def run_actor_training(samples, critic, actor):
    for sample in samples:
        cur_state, action, reward, new_state, _ = sample
        predicted_action = actor.predict(cur_state)
        action_gradients = critic.action_gradients(cur_state, predicted_action)[0]
        actor.train(cur_state, action_gradients)


def run_critic_training(samples, critic, actor):
    for sample in samples:
        cur_state, action, reward, new_state, done = sample
        if not done:
            target_action = actor.target_predict(new_state)
            future_reward = critic.target_predict(new_state, target_action)[0][0]
            reward += critic.gamma * future_reward
            reward = reward.numpy()
        critic.model.fit([cur_state, action], np.asarray([reward]), verbose=0)


def run_training(actor, critic, buffer):
    batch_size = 32
    if len(buffer.memory) < batch_size:
        return

    samples = random.sample(buffer.memory, batch_size)
    run_critic_training(samples, critic, actor)
    run_actor_training(samples, critic, actor)


def main():
    env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")

    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = .995
    gamma = .95
    tau = .125

    actor = Actor(env, learning_rate, epsilon, epsilon_decay, gamma, tau)
    critic = Critic(env, learning_rate, epsilon, epsilon_decay, gamma, tau)
    buffer = Buffer()

    num_trials = 10000
    trial_len = 500

    # env.render()
    for i in range(num_trials):

        print("Epoch {}".format(i))
        cur_state = env.reset()

        for j in range(trial_len):

            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor.act(cur_state)
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, _ = env.step(action[0])
            new_state = new_state.reshape((1, env.observation_space.shape[0]))
            buffer.remember(cur_state, action, reward, new_state, done)

            run_training(actor, critic, buffer)

            cur_state = new_state

            actor.update_target_network()
            critic.update_target_network()

            if done:
                break


if __name__ == "__main__":
    main()
