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
    def __init__(self, env, lr, gamma, tau):
        self.learning_rate = lr
        self.gamma = gamma
        self.tau = tau

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.model = self.create_critic_network()
        self.target_model = self.create_critic_network()

    def create_critic_network(self):
        state = Input(shape=(self.state_dim,), name='critic_state_input')
        #version 1
        #state_hidden_1 = Dense(400, activation='relu', name='critic_state_hidden_1')(state)

        #version 2
        state_hidden_1 = Dense(400, use_bias=False)(state)
        state_hidden_2 = BatchNormalization()(state_hidden_1)
        state_hidden_3 = Activation('relu')(state_hidden_2)

        state_hidden_2 = Dense(300, use_bias=False, name='critic_state_hidden_2')(state_hidden_3)

        action = Input(shape=(self.action_dim,), name='critic_action_input')
        action_hidden_1 = Dense(300, name='critic_action_hidden_1')(action)

        merged = Add(name='critic_merged')([state_hidden_2, action_hidden_1])
        merged_hidden_1 = Activation('relu', name='critic_merged_hidden_1')(merged)

        weights_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output = Dense(1, kernel_initializer=weights_initializer, bias_initializer=weights_initializer,
                       name='critic_output')(merged_hidden_1)
        model = Model(inputs=[state, action], outputs=output)

        optimizer = Adam(lr=self.learning_rate)
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
            out = self.model([state, action])
        return tape.gradient(out, action)

    @tf.function
    def update_target_network(self):
        for i in range(len(self.target_model.trainable_weights)):
            self.target_model.trainable_weights[i].assign(
                tf.math.multiply(self.model.trainable_weights[i], self.tau)
                + tf.math.multiply(self.target_model.trainable_weights[i], 1. - self.tau))
