import tensorflow as tf
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

class MeanAggregator():
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):

        self.name = name
        self.bias = bias
        self.act = act
        self.concat = concat
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_layer = tf.keras.layers.Dense(self.output_dim, use_bias=self.bias, name=name+'aggN')
        self.self_layer = tf.keras.layers.Dense(self.output_dim, use_bias=self.bias, name=name+'aggS')

    # hop used for names
    def __call__(self, inputs, hop, layer):
        self_vecs, neigh_vecs = inputs
        if hop == 0:
            neigh_means = tf.keras.layers.AveragePooling1D(FLAGS.samples_2, name=self.name+'hop'+str(hop)+'neigh_feat_avg')(neigh_vecs)
        else:
            neigh_means = neigh_vecs
        neigh_means = tf.keras.layers.Flatten(name=self.name+'hop'+str(hop)+'neigh_feat_flatten')(neigh_means)
        self_vecs = tf.keras.layers.Flatten(name=self.name+'hop'+str(hop)+'self_feat_flatten')(self_vecs)
        from_neighs = self.neigh_layer(neigh_means)
        from_self = self.self_layer(self_vecs)
        retval = tf.keras.layers.concatenate([from_self, from_neighs], name=self.name+'hop'+str(hop)+'aggconcat')
        return self.act(retval)