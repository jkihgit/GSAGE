import tensorflow as tf

class MeanAggregator():
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):

        self.bias = bias
        self.act = act
        self.concat = concat
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_layer = tf.keras.layers.Dense(self.output_dim, input_shape=(self.neigh_input_dim,), activation=self.act, use_bias=self.bias, name=name+'aggN')
        self.self_layer = tf.keras.layers.Dense(self.output_dim, input_shape=(self.input_dim,), activation=self.act, use_bias=self.bias, name=name+'aggS')

    def __call__(self, inputs):
        self_vecs, neigh_vecs = inputs
        l = neigh_vecs.shape[1]
        neigh_means = tf.keras.layers.AveragePooling1D(l)(neigh_vecs)
        neigh_means = tf.keras.layers.Flatten()(neigh_means)
        from_neighs = self.neigh_layer(neigh_means)
        from_self = self.self_layer(self_vecs)
        return tf.keras.layers.concatenate([from_self, from_neighs])

    def weights(self):
        return self.self_layer.weights + self.neigh_layer.weights