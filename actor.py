import tensorflow as tf

tf.keras.backend.set_floatx('float64')

Model = tf.keras.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation

Adam = tf.keras.optimizers.Adam


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.model = self.create_actor_network()
        self.target_model = self.create_actor_network()

        self.optimizer = Adam(learning_rate=self.learning_rate)

    def create_actor_network(self):
        inputs = Input(shape=(self.state_dim,))
        hidden_1 = Dense(400, input_shape=(self.state_dim,), use_bias=False)(inputs)
        hidden_2 = BatchNormalization()(hidden_1)
        hidden_3 = Activation('relu')(hidden_2)
        hidden_4 = Dense(300, use_bias=False)(hidden_3)
        hidden_5 = BatchNormalization()(hidden_4)
        hidden_6 = Activation('relu')(hidden_5)

        weights_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=weights_initializer,
                    bias_initializer=weights_initializer)(hidden_6)
        scaled_out = tf.math.multiply(out, self.action_bound)

        return Model(inputs=inputs, outputs=scaled_out)

    def train(self, inputs, action_gradient):
        with tf.GradientTape() as tape:
            loss = self.model(inputs)
        unnormalized_actor_gradients = tape.gradient(loss, self.model.trainable_variables, -action_gradient)
        actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), unnormalized_actor_gradients)) # tf.div --> tf.math.divide

        self.optimize(actor_gradients)

    @tf.function
    def optimize(self, actor_gradients):
        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_variables))

    @tf.function
    def predict(self, inputs):
        return self.model(inputs)

    @tf.function
    def target_predict(self, inputs):
        return self.target_model(inputs)

    @tf.function
    def update_target_network(self):
        for i, _ in enumerate(self.target_model.trainable_variables):
            self.target_model.trainable_variables[i].assign(
            tf.math.multiply(self.model.trainable_variables[i], self.tau)
            + tf.math.multiply(self.target_model.trainable_variables[i], 1. - self.tau))


