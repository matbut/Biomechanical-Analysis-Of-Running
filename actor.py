import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

Model = tf.keras.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation

Adam = tf.keras.optimizers.Adam


class Actor:
    def __init__(self, env, lr, eps, eps_decay, tau):
        self.learning_rate = lr
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.tau = tau

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.optimizer = Adam(learning_rate=self.learning_rate)

        self.model = self.create_actor_network()
        self.target_model = self.create_actor_network()

    def create_actor_network(self):
        inputs = Input(shape=(self.state_dim,), name='actor_inputs')
        # version 1
        # hidden_1 = Dense(400, activation='relu', name='actor_hidden_1')(inputs)
        # hidden_2 = Dense(300, activation='relu', name='actor_hidden_2')(hidden_1)

        # version 2
        hidden_1 = Dense(400, use_bias=False)(inputs)
        hidden_2 = BatchNormalization()(hidden_1)
        hidden_3 = Activation('relu')(hidden_2)
        hidden_4 = Dense(300, use_bias=False)(hidden_3)
        hidden_5 = BatchNormalization()(hidden_4)
        hidden_6 = Activation('relu')(hidden_5)

        weights_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output = Dense(self.action_dim, activation='tanh',
                       kernel_initializer=weights_initializer, bias_initializer=weights_initializer,
                       name='actor_output')(hidden_6)

        # scaling output range [-1, 1] to action range [env.action_space.low, self.env.action_space.high]
        scaled_output = Scaler(self.env, name='actor_scaled_output')(output)

        model = Model(inputs=inputs, outputs=scaled_output)
        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.predict(state).numpy()[0]

    @tf.function
    def train(self, cur_state_batch, action_gradients_batch):
        batch_size = cur_state_batch.shape[0]
        with tf.GradientTape() as tape:
            out = self.model(cur_state_batch, training=True)
        accumulated_actor_gradients_batch = tape.gradient(out, self.model.trainable_weights, -action_gradients_batch)
        actor_gradients_batch = list(map(lambda t: t / batch_size, accumulated_actor_gradients_batch))
        self.optimizer.apply_gradients(zip(actor_gradients_batch, self.model.trainable_weights))

    @tf.function
    def predict(self, inputs):
        return self.model(inputs)

    @tf.function
    def target_predict(self, inputs):
        return self.target_model(inputs)

    @tf.function
    def update_target_network(self):
        for i in range(len(self.target_model.trainable_weights)):
            self.target_model.trainable_weights[i].assign(
                tf.math.multiply(self.model.trainable_weights[i], self.tau)
                + tf.math.multiply(self.target_model.trainable_weights[i], 1. - self.tau))


class Scaler(tf.keras.layers.Layer):

    def __init__(self, env, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        action_center = (env.action_space.high - env.action_space.low) / 2
        action_amplitude = (env.action_space.high + env.action_space.low) / 2

        action_dim = env.action_space.shape[0]

        self.action_center = tf.constant(action_center.reshape((action_dim,)), dtype=tf.float64)
        self.action_amplitude = tf.constant(action_amplitude.reshape((action_dim,)), dtype=tf.float64)

    def call(self, inputs, **kwargs):
        return tf.map_fn(lambda t: tf.add(tf.multiply(t, self.action_amplitude), self.action_center), inputs)

