import tensorflow as tf

tf.keras.backend.set_floatx('float64')

Model = tf.keras.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation

Adam = tf.keras.optimizers.Adam


class Critic:
    def __init__(self, state_dim, action_dim, learning_rate, tau, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.model = self.create_critic_network()
        self.target_model = self.create_critic_network()

        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.loss = lambda q, s, a: tf.keras.losses.MSE(self.model([s, a], training=True), q)

    def create_critic_network(self):
        state = Input(shape=(self.state_dim,))
        action = Input(shape=(self.action_dim,))

        hidden_1 = Dense(400, input_shape=(self.state_dim,), use_bias=False)(state)
        hidden_2 = BatchNormalization()(hidden_1)
        hidden_3 = Activation('relu')(hidden_2)

        l_1 = Dense(300)
        l_2 = Dense(300)
        tmp_1 = l_1(hidden_3)
        tmp_2 = l_2(action)

        w1 = l_1.get_weights()[0]
        w2 = l_2.get_weights()[0]
        b2 = l_2.get_weights()[1]

        hidden_6 = Activation('relu')(tf.linalg.matmul(hidden_3, w1) + tf.linalg.matmul(action, w2) + b2)

        weights_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        out = Dense(1, kernel_initializer=weights_initializer, bias_initializer=weights_initializer)(hidden_6)

        return Model(inputs=[state, action], outputs=out)

    @tf.function
    def train(self, state, action, predicted_q_value):
        self.optimizer.minimize(lambda: self.loss(predicted_q_value, state, action),
                                var_list=self.model.trainable_variables)
        return self.model([state, action], training=True)

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
            out = self.model([state, action], training=True)
        return tape.gradient(out, action)

    @tf.function
    def update_target_network(self):
        for i, _ in enumerate(self.target_model.trainable_variables):
            self.target_model.trainable_variables[i].assign(
                tf.math.multiply(self.model.trainable_variables[i], self.tau)
                + tf.math.multiply(self.target_model.trainable_variables[i], 1. - self.tau))
