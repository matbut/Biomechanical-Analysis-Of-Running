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
    def __init__(self, env, lr, eps, eps_decay, gamma, tau):
        self.learning_rate = lr
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.tau = tau

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.model = self.create_actor_network()
        self.target_model = self.create_actor_network()

        self.optimizer = Adam(learning_rate=self.learning_rate)

    def create_actor_network(self):
        inputs = Input(shape=(self.state_dim,))
        hidden_1 = Dense(400, activation='relu')(inputs)
        hidden_2 = Dense(300, activation='relu')(hidden_1)
        hidden_3 = Dense(400, activation='relu')(hidden_2)
        output = Dense(self.action_dim, activation='relu', )(hidden_3)

        model = Model(inputs=inputs, outputs=output)
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
        actor_gradients = tape.gradient(out, self.model.trainable_variables, -action_gradients)
        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_variables))

    @tf.function
    def predict(self, inputs):
        return self.model(inputs)

    @tf.function
    def target_predict(self, inputs):
        return self.target_model(inputs)

    @tf.function
    def update_target_network(self):
        for i in range(len(self.target_model.trainable_variables)):
            self.target_model.trainable_variables[i].assign(
            tf.math.multiply(self.model.trainable_variables[i], self.tau)
            + tf.math.multiply(self.target_model.trainable_variables[i], 1. - self.tau))


