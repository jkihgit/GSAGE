core dump right after printing "Epoch 1/100" while running
basic_string::append()
due to
std::length_error

when running reddit dataset

no issues when running pubmed dataset


likely a OOM issue caused by literal OOM or 32bit restricton from python or numpy (probably numpy)
    correct: 32bit max len is not enough for the model
    problem: python be running in 64bit

    oh yeah this thing
        https://stackoverflow.com/questions/34128872/google-protobuf-maximum-size/34186672
        that i ran in to
        last time

Looks like model.fit_generator() can fix this

Problem: slow

https://github.com/tensorflow/tensorflow/issues/32104
bug in RC2
fixed after using workaround in the link (forbid eager mode)

now runs reddit
    but fails to converge (stable @ 5% accuracy)
    wot

    pubmed still fine

    check auto migration warning (tf.nn.embedding_lookup() only provides partition strategy div. Partition strategy mod used by code. Manual check required)
        used tf.Print() to check embedding_lookup() output
        same output as graphsage/fresh
        (checked ids head & adj_list head)

    tips to avoid gradient explosion
        gradient clipping / normalization (done)
        decaying LR (done, we're gold)
        large (0.01 or 0.001) epsilon
        using a more modern version of Adam
