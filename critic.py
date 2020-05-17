import tensorflow as tf

tf.keras.backend.set_floatx('float64')

Model = tf.keras.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation

Adam = tf.keras.optimizers.Adam


class Critic:
    def __init__(self, env, lr, eps, eps_decay, gamma, tau):
        self.learning_rate = lr
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.gamma = gamma
        self.tau = tau

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.model = self.create_critic_network()
        self.target_model = self.create_critic_network()

    def create_critic_network(self):
        state = Input(shape=(self.state_dim,))
        state_hidden_1 = Dense(400, activation='relu')(state)
        state_hidden_2 = Dense(300, activation='relu')(state_hidden_1)

        action = Input(shape=(self.action_dim,))
        action_hidden_1 = Dense(300)(action)

        merged = Add()([state_hidden_2, action_hidden_1])
        merged_hidden_1 = Dense(300, activation='relu')(merged)

        output = Dense(1, activation='relu')(merged_hidden_1)
        model = Model(inputs=[state, action], outputs=output)

        optimizer = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    @tf.function
    def predict(self, state, action):
        return self.model([state, action])

    @tf.function
    def target_predict(self, state, action):
        return self.target_model([state, action])

    @tf.function
    def action_gradients(self, state, action):
        with tf.GradientTape() as tape:
            tape.watch(action)
            tape.watch(state)
            out = self.model([state, action], training=True)
        return tape.gradient(out, action)

    @tf.function
    def update_target_network(self):
        for i in range(len(self.target_model.trainable_variables)):
            self.target_model.trainable_variables[i].assign(
                tf.math.multiply(self.model.trainable_variables[i], self.tau)
                + tf.math.multiply(self.target_model.trainable_variables[i], 1. - self.tau))
