import gym
import pybullet_envs  # do not delete, necessary to make gym env
import time

env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')
env.render(mode='human')
env.reset()

dt = 1. / 240.

action = env.env.action_space.sample()
while True:
    time.sleep(dt)
    action = env.env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        env.reset()
