import numpy as np
import tensorflow as tf

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
from replay_buffer import ReplayBuffer


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Agent Training
# ===========================

def train(env, args, actor, critic, actor_noise):

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render(mode='human')

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.state_dim))) + actor_noise()
            print("j=",j)
            print("s=",s)
            print("a=", a)

            s2, r, terminal, info = env.step(a[0])
            print("s2=", s2)

            replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r,
                              terminal, np.reshape(s2, (actor.state_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            #FIXME something goes wrong after we reach reach j=args['minibatch_size'] and go inside IF body below
            # after a while predicted action is a Tensor od NaNs
            # seems like there is something wrong with the Actor.train? (probably gradients?)
            # sometimes weights in actor model become NaNs or really large values ~ 10^100
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.target_predict(
                    s2_batch, actor.target_predict(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        print("1")
                        y_i.append(tf.convert_to_tensor([r_batch[k]])) # initially: y_i.append(r_batch[k])
                    else:
                        print("2")
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                y_i_reshaped = np.reshape(y_i, (int(args['minibatch_size']), 1))
                predicted_q_value = critic.train(s_batch, a_batch, y_i_reshaped)

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                ## FIXME: after running actor.train a few times weights in actor become NaNs

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()


            s = s2
            ep_reward += r

            print("reward:", ep_reward)

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break
