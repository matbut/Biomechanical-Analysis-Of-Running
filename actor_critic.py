import os

import gym
import numpy as np
import time

import pybullet_envs  # do not delete, necessary to make gym env

from actor import Actor
from buffer import Buffer
from critic import Critic

batch_size = 64

restore_epoch = -1  # set to -1 when training without restoring weights
make_checkpoints = True  # whether to save models' weights so that they could be restored later
checkpoints_freq = 100
hyperparam_freq = 10
render = False  # whether to display humanoid model
training_num = 3
restore_training_num = 2

checkpoint_path = '../checkpoints/training_{training_num}/{model}/{epoch:04d}.ckpt'

hyperparam_path = '../checkpoints/training_{}/reward/reward.csv'.format(training_num)
hyperparam_dir = os.path.dirname(hyperparam_path)

if not os.path.exists(hyperparam_dir):
    os.makedirs(hyperparam_dir)
    f = open(hyperparam_path, 'w')
    f.close()


def run_actor_training(samples, critic, actor):
    cur_state_batch, _, _, _, _ = samples
    predicted_action_batch = actor.predict(cur_state_batch)
    action_gradients_batch = critic.action_gradients(cur_state_batch, predicted_action_batch)
    actor.train(cur_state_batch, action_gradients_batch)


def run_critic_training(samples, critic, actor):
    cur_state_batch, action_batch, reward_batch, new_state_batch, done_batch = samples

    target_action_batch = actor.target_predict(new_state_batch)
    future_reward_batch = critic.target_predict(new_state_batch, target_action_batch).numpy()[:, 0]

    reward_batch[~done_batch] += critic.gamma * future_reward_batch[~done_batch]

    critic.model.fit([cur_state_batch, action_batch], [reward_batch], verbose=0)

    return np.max(future_reward_batch)


def run_training(actor, critic, buffer):
    samples = buffer.get_random_batch(batch_size)
    max_reward = run_critic_training(samples, critic, actor)
    run_actor_training(samples, critic, actor)
    return max_reward


def main():
    env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')

    actor_learning_rate = 0.00001
    critic_learning_rate = 0.0001
    gamma = .99#.95
    tau = .001#.125

    actor = Actor(env, actor_learning_rate, tau)
    critic = Critic(env, critic_learning_rate, gamma, tau)
    buffer = Buffer()

    if restore_epoch >= 0:
        actor.model.load_weights(checkpoint_path.format(training_num=restore_training_num, model='actor_model', epoch=restore_epoch))
        actor.target_model.load_weights(checkpoint_path.format(training_num=restore_training_num, model='actor_target', epoch=restore_epoch))
        critic.model.load_weights(checkpoint_path.format(training_num=restore_training_num, model='critic_model', epoch=restore_epoch))
        critic.target_model.load_weights(checkpoint_path.format(training_num=restore_training_num, model='critic_target', epoch=restore_epoch))

    num_epochs = 50000
    epoch_len = 1000

    if render:
        env.render()

    for epoch in range(num_epochs):

        print("Epoch {}".format(epoch))
        cur_state = env.reset()

        max_train_reward = 0
        max_reward = 0

        for j in range(epoch_len):

            action = actor.act(cur_state.reshape((1, env.observation_space.shape[0])))

            new_state, reward, done, _ = env.step(action.reshape((1, env.action_space.shape[0]))[0])
            buffer.remember(cur_state, action, reward, new_state, done)

            max_reward = max(max_reward, reward)

            if len(buffer.memory) > batch_size:
                avg_reward = run_training(actor, critic, buffer)
                max_train_reward = max(max_train_reward, avg_reward)

                cur_state = new_state

                actor.update_target_network()
                critic.update_target_network()

            if done:
                break

        print('Max_reward {}\nAvg_max_train_reward {}\n'.format(max_reward, max_train_reward / epoch_len))

        if make_checkpoints and (epoch % checkpoints_freq == 0):
            actor.model.save_weights(checkpoint_path.format(training_num=training_num, model='actor_model', epoch=epoch))
            actor.target_model.save_weights(checkpoint_path.format(training_num=training_num, model='actor_target', epoch=epoch))
            critic.model.save_weights(checkpoint_path.format(training_num=training_num, model='critic_model', epoch=epoch))
            critic.target_model.save_weights(checkpoint_path.format(training_num=training_num, model='critic_target', epoch=epoch))

        if make_checkpoints and (epoch % hyperparam_freq == 0):
            file_path = hyperparam_path.format(training_num=training_num)
            with open(file_path, 'a') as f:
                f.write('{},{}\n'.format(max_reward, max_train_reward / epoch_len))


def test():
    env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')
    env.render()

    actor_learning_rate = 0.00001
    critic_learning_rate = 0.0001
    gamma = .99  # .95
    tau = .001  # .125

    actor = Actor(env, actor_learning_rate, tau)
    critic = Critic(env, critic_learning_rate, gamma, tau)
    actor.model.load_weights(
        checkpoint_path.format(training_num=restore_training_num, model='actor_model', epoch=restore_epoch))
    actor.target_model.load_weights(
        checkpoint_path.format(training_num=restore_training_num, model='actor_target', epoch=restore_epoch))
    critic.model.load_weights(
        checkpoint_path.format(training_num=restore_training_num, model='critic_model', epoch=restore_epoch))
    critic.target_model.load_weights(
        checkpoint_path.format(training_num=restore_training_num, model='critic_target', epoch=restore_epoch))

    cur_state = env.reset()

    for _ in range(100000):
        action = actor.act(cur_state.reshape((1, env.observation_space.shape[0])))
        new_state, reward, done, _ = env.step(action.reshape((1, env.action_space.shape[0]))[0])
        cur_state = new_state
        time.sleep(.1)
        if done:
            break


if __name__ == "__main__":
    main()
    #test()
