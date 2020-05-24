import os

import gym

import random

import numpy as np
import pybullet_envs  # do not delete, necessary to make gym env

from actor import Actor
from buffer import Buffer
from critic import Critic

# TODO enable processing whole batch at once

batch_size = 32

restore_epoch = 4  # set to -1 when training without restoring weights
make_checkpoints = True  # whether to save models' weights so that they could be restored later
render = False  # whether to display humanoid model

checkpoint_path = "../checkpoints/training_1/{model}/{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


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
            future_reward = critic.target_predict(new_state, target_action)[0][0].numpy()
            reward += critic.gamma * future_reward
        critic.model.fit([cur_state, action], np.array([reward]), verbose=0)


def run_training(actor, critic, buffer):
    samples = random.sample(buffer.memory, batch_size)
    run_critic_training(samples, critic, actor)
    run_actor_training(samples, critic, actor)


def main():
    env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')

    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = .995
    gamma = .95
    tau = .125

    actor = Actor(env, learning_rate, epsilon, epsilon_decay, gamma, tau)
    critic = Critic(env, learning_rate, epsilon, epsilon_decay, gamma, tau)
    buffer = Buffer()

    if restore_epoch >= 0:
        actor.model.load_weights(checkpoint_path.format(model='actor_model', epoch=restore_epoch))
        actor.target_model.load_weights(checkpoint_path.format(model='actor_target', epoch=restore_epoch))
        critic.model.load_weights(checkpoint_path.format(model='critic_model', epoch=restore_epoch))
        critic.target_model.load_weights(checkpoint_path.format(model='critic_target', epoch=restore_epoch))

    num_epochs = 50000
    epoch_len = 500

    if render:
        env.render()

    for epoch in range(num_epochs):

        print("Epoch {}".format(epoch))
        cur_state = env.reset()

        max_reward = 0

        for j in range(epoch_len):

            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor.act(cur_state)
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, _ = env.step(action[0])
            new_state = new_state.reshape((1, env.observation_space.shape[0]))
            buffer.remember(cur_state, action, reward, new_state, done)

            if len(buffer.memory) > batch_size:
                run_training(actor, critic, buffer)

                cur_state = new_state

                actor.update_target_network()
                critic.update_target_network()

                max_reward = max(max_reward, reward)

            if done:
                break

        print('Max_reward {}'.format(max_reward))

        if make_checkpoints:
            actor.model.save_weights(checkpoint_path.format(model='actor_model', epoch=epoch))
            actor.target_model.save_weights(checkpoint_path.format(model='actor_target', epoch=epoch))
            critic.model.save_weights(checkpoint_path.format(model='critic_model', epoch=epoch))
            critic.target_model.save_weights(checkpoint_path.format(model='critic_target', epoch=epoch))


if __name__ == "__main__":
    main()
