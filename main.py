import argparse
import gym
import time

import numpy as np
import tensorflow as tf
import pprint as pp

from actor import Actor
from actor_critic import OrnsteinUhlenbeckActionNoise, train
from critic import Critic
from gym import wrappers


import pybullet_envs  # do not delete, necessary to make gym env

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(args):

    env = gym.make(args['env'])
    np.random.seed(int(args['random_seed']))
    tf.random.set_seed(int(args['random_seed']))
    env.seed(int(args['random_seed']))

    if args['render_env']:
        env.render(mode='human')
        env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    # print(env.action_space.high)
    # print(env.action_space.low)
    # assert (env.action_space.high == -env.action_space.low)

    actor = Actor(state_dim, action_dim, action_bound,
                         float(args['actor_lr']), float(args['tau']),
                         int(args['minibatch_size']))

    critic = Critic(state_dim, action_dim,
                           float(args['critic_lr']), float(args['tau']),
                           float(args['gamma']),)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    if args['use_gym_monitor']:
        if not args['render_env']:
            env = wrappers.Monitor(
                env, args['monitor_dir'], video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, args['monitor_dir'], force=True)

    train(env, args, actor, critic, actor_noise)

    # if args['use_gym_monitor']:
    #    env.monitor.close()

    env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')
    env.render(mode='human')
    s = env.reset()

    for i in range(10000):
        time.sleep(1. / 30.)
        a = actor.predict(np.reshape(s, (1, actor.state_dim)))
        s, r, done, info = env.step(a[0])
        if done:
            print(i)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
