import tensorflow as tf

# def MeanAggregator(input_dim, output_dim, batch_size=None, bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
#     neigh_input_dim = input_dim
#     inputs = tf.keras.Input(shape=(2,None,input_dim), batch_size=batch_size)
#     neigh_layer = tf.keras.layers.Dense(output_dim, input_shape=(neigh_input_dim,), activation=act, use_bias=bias, name=name+'aggN')
#     self_layer = tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), activation=act, use_bias=bias, name=name+'aggS')
#     neigh_vecs = inputs[:,1]
#     self_vecs = inputs[:,0]
#     neigh_means = tf.math.reduce_mean(neigh_vecs, 1, name=name+'neigh_feat_avg')
#     self_vecs = tf.math.reduce_sum(self_vecs, 1, name=name+'self_feat_flatten')
#     from_neighs = neigh_layer(neigh_means)
#     from_self = self_layer(self_vecs)
#     retval = tf.keras.layers.concatenate([from_self, from_neighs], name=name+'aggconcat')
#     return tf.keras.models.Model(inputs, retval, name=name)

# this is way too complex for something that should be simple
# try this approach instead
# https://stackoverflow.com/questions/49875127/share-weights-between-two-dense-layers-in-keras

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
        self.neigh_layer = tf.keras.layers.Dense(self.output_dim, use_bias=self.bias, kernel_initializer='zeros', name=name+'aggN')
        self.self_layer = tf.keras.layers.Dense(self.output_dim, use_bias=self.bias, kernel_initializer='zeros', name=name+'aggS')
        # self.neigh_layer = tf.keras.layers.Dense(self.output_dim, input_shape=(self.neigh_input_dim,), use_bias=self.bias, name=name+'aggN')
        # self.self_layer = tf.keras.layers.Dense(self.output_dim, input_shape=(self.input_dim,), use_bias=self.bias, name=name+'aggS')

    # hop used for names
    def __call__(self, inputs, hop):
        self_vecs, neigh_vecs = inputs
        l = [neigh_vecs.shape[1]]
        neigh_means = tf.keras.layers.AveragePooling1D(l, name=self.name+'hop'+str(hop)+'neigh_feat_avg')(neigh_vecs)
        neigh_means = tf.keras.layers.Flatten(name=self.name+'hop'+str(hop)+'neigh_feat_flatten')(neigh_means)
        self_vecs = tf.keras.layers.Flatten(name=self.name+'hop'+str(hop)+'self_feat_flatten')(self_vecs)
        from_neighs = self.neigh_layer(neigh_means)
        from_self = self.self_layer(self_vecs)
        retval = tf.keras.layers.concatenate([from_self, from_neighs], name=self.name+'hop'+str(hop)+'aggconcat')
        return self.act(retval)