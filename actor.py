import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

Model = tf.keras.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add
Multiply = tf.keras.layers.Multiply
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation

Adam = tf.keras.optimizers.Adam


class Actor:
    def __init__(self, env, lr, eps, eps_decay, gamma, tau):
        self.learning_rate = lr
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.tau = tau

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        action_center = (self.env.action_space.high - self.env.action_space.low) / 2
        action_amplitude = (self.env.action_space.high + self.env.action_space.low) / 2
        self.action_center = tf.convert_to_tensor(action_center.reshape((1, 36)))
        self.action_amplitude = tf.convert_to_tensor(action_amplitude.reshape((1, 36)))

        self.model = self.create_actor_network()
        self.target_model = self.create_actor_network()

        self.optimizer = Adam(learning_rate=self.learning_rate)

    def create_actor_network(self):
        inputs = Input(shape=(self.state_dim,), name='actor_inputs')
        #version 1
        #hidden_1 = Dense(400, activation='relu', name='actor_hidden_1')(inputs)
        #hidden_2 = Dense(300, activation='relu', name='actor_hidden_2')(hidden_1)

        #version 2
        hidden_1 = Dense(400, use_bias=False)(inputs)
        hidden_2 = BatchNormalization()(hidden_1)
        hidden_3 = Activation('relu')(hidden_2)
        hidden_4 = Dense(300, use_bias=False)(hidden_3)
        hidden_5 = BatchNormalization()(hidden_4)
        hidden_6 = Activation('relu')(hidden_5)

        weights_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output = Dense(self.action_dim, activation='tanh', kernel_initializer=weights_initializer,
                       bias_initializer=weights_initializer, name='actor_output')(hidden_6)

        #scaling output range [-1, 1] to action range [env.action_space.low, self.env.action_space.high]
        scaled_output = Multiply(name='actor_scaled_output')([output, self.action_amplitude])
        translated_output = Add(name='actor_translated_output')([scaled_output, self.action_center])

        model = Model(inputs=inputs, outputs=translated_output)
        optimizer = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.predict(state).numpy()

    @tf.function
    def train(self, cur_state, action_gradients):
        with tf.GradientTape() as tape:
            out = tf.reshape(self.model(cur_state, training=True), (1, self.action_dim))
        actor_gradients = tape.gradient(out, self.model.trainable_weights, -action_gradients)
        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_weights))

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


