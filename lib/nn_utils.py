import tensorflow as tf


''' NN utils '''


def lecun_truncated_normal_initializer(shape, dtype=None, partition_info=None):
    """
    自定义 LeCun Truncated Normal 初始化器
    截断正态分布，均值为0，标准差为sqrt(1 / fan_in)
    """
    fan_in = shape[0]  # 输入的特征数
    stddev = tf.sqrt(1 / fan_in)  # LeCun 的标准差计算公式
    return tf.random.truncated_normal(shape, mean=0.0, stddev=stddev, dtype=dtype)


def make_nn(output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = [tf.keras.layers.Dense(
        h, activation=tf.nn.relu, dtype=tf.float32,
        # kernel_initializer=lecun_truncated_normal_initializer
    )
              for h in hidden_sizes]
    layers.append(tf.keras.layers.Dense(
        output_size, dtype=tf.float32,
        # kernel_initializer=lecun_truncated_normal_initializer,
    ))
    return tf.keras.Sequential(layers)


def make_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of
          one 1d-convolutional layer that utilizes temporal dependences,
          fully connected network

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """
    cnn_layer = [tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size,
                                        padding="same", dtype=tf.float32)]
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes[1:]]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(cnn_layer + layers)


def make_2d_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Creates fully convolutional neural network.
        Used as CNN preprocessor for image data (HMNIST, SPRITES)

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """
    layers = [tf.keras.layers.Conv2D(h, kernel_size=kernel_size, padding="same",
                                     activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes + [output_size]]
    return tf.keras.Sequential(layers)

