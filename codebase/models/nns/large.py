import tensorflow as tf
from codebase.args import args
from codebase.models.extra_layers import leaky_relu, noise
from tensorbayes.layers import dense, conv2d, avg_pool, max_pool, batch_norm, instance_norm
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.layers.core import dropout

kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.zeros_initializer()

def classifier(x, phase, gen_phase=0, gen_trim=1, enc_phase=0, enc_trim=3, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            preprocess = instance_norm if args.inorm else tf.identity
            layout = [
                (preprocess, (), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 5, 1), dict(padding='VALID')),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (conv2d, (192, 5, 1), {}),
                (conv2d, (192, 5, 1), {}),
                (conv2d, (192, 5, 1), dict(padding='VALID')),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (tf.reshape, ([-1, 25*192],), {}),
                (dense, (2048,), dict(bn=False)),
                (dense, (args.Y,), dict(activation=None))
            ]

            if enc_phase:
                start = 0
                end = len(layout) - enc_trim
            elif gen_phase:
                start = len(layout) - enc_trim
                end = len(layout) - gen_trim
            else:
                start = len(layout) - gen_trim
                end = len(layout)

            for i in xrange(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)

    return x

def real_feature_discriminator(x, phase, C=1, reuse=None):
    with tf.variable_scope('disc_real', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu): # Switch to leaky?

            x = dense(x, 500)
            x = dense(x, 100)
            x = dense(x, C, activation=None)

    return x

def trg_generator(z):
    with tf.variable_scope('trg_gen', reuse=tf.AUTO_REUSE):
        h = bn_dense(z, 2*2*1024, bias=False)    # (B,100)->(B,2*2*1024)
        h = tf.reshape(h, [-1, 2, 2, 1024])      # (B,2*2*1024)->(B,2,2,1024)
        h = bn_trans_convlayer(h, 5, 2, 512, bias=False)    # (B,2,2,1024)->(B,4,4,512)
        h = bn_trans_convlayer(h, 5, 2, 256, bias=False)    # (B,4,4,512)->(B,8,8,256)
        h = bn_trans_convlayer(h, 5, 2, 256, bias=False)    # (B,8,8,256)->(B,16,16,256)
        h = bn_trans_convlayer(h, 5, 2, 128, bias=False)    # (B,16,16,256)->(B,32,32,128)
        out = trans_convlayer(h, 5, 1, 3, nonlinearity=tf.nn.tanh)  # (B,32,32,128)->(B,32,32,3)
    return out

# Utility functions
def bn_dense(inputs, units, bias=True, nonlinearity=tf.nn.leaky_relu):
    return tf.contrib.layers.batch_norm(
        tf.layers.dense(inputs=inputs, units=units, kernel_initializer=kernel_initializer,
                    use_bias=bias, bias_initializer=bias_initializer),
        activation_fn=nonlinearity, fused=True)

def trans_convlayer(inputs, kernel_size, strides, filters, bias=True, padding='same', nonlinearity=tf.nn.leaky_relu):
    return tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                            strides=strides, padding=padding, activation=nonlinearity,
                            use_bias=bias, bias_initializer=bias_initializer)

def bn_trans_convlayer(inputs, kernel_size, strides, filters, bias=True, padding='same', nonlinearity=tf.nn.leaky_relu):
    return tf.contrib.layers.batch_norm(
        tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                strides=strides, padding=padding, use_bias=bias, bias_initializer=bias_initializer),
        activation_fn=nonlinearity, fused=True)
