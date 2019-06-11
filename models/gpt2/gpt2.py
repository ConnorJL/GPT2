import numpy as np
import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        if params["precision"] == "bfloat16":
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, scale=1.0, params=None):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        if scope=="c_attn":
            scale = 1.0
        if params["precision"] == "bfloat16":
            w = tf.get_variable('w', [1, nx, nf], initializer=ScaledNormalInitializer(stddev=w_init_stdev, scale=scale, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            w = tf.get_variable('w', [1, nx, nf], initializer=ScaledNormalInitializer(stddev=w_init_stdev, scale=scale))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, params, train=False):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, params["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)

        w = dropout(w, params["attn_dropout"], train)

        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, params=params)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, scale=params["scale"], params=params)
        a = dropout(a, params["res_dropout"], train)
        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, scale=params["scale"], params=params)
        h2 = dropout(h2, params["res_dropout"], train)
        return h2


def block(x, scope, *, past, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1', params=params), 'attn', nx, past=past, params=params, train=train)
        x = x + a
        m = mlp(norm(x, 'ln_2', params=params), 'mlp', nx*4, params=params, train=train)
        x = x + m
        return x, present

def past_shape(*, params, batch_size=None, sequence=None):
    return [batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

def _assert_float_dtype(dtype):
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype

# Initializer that scales the parameters as in the original paper
class ScaledNormalInitializer(tf.keras.initializers.Initializer):
    #def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, scale=1.0):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=tf.float32, scale=1.0):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.dtype = _assert_float_dtype(tf.dtypes.as_dtype(dtype))
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return tf.math.multiply(tf.random.normal(
            shape, self.mean, self.stddev, dtype, seed=self.seed), self.scale)

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
            "dtype": self.dtype.name,
            "scale": self.scale
        }


def model(X, params, labels=None, past=None, scope='model', reuse=False, train=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if params["precision"] == "bfloat16":
            wpe = tf.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.bfloat16), dtype=tf.bfloat16)
            wte = tf.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.bfloat16), dtype=tf.bfloat16)

        else:
            wpe = tf.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                                initializer=tf.random_normal_initializer(stddev=0.01))
            wte = tf.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
                                initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]

        wpe = dropout(wpe, params["embed_dropout"], train)
        wte = dropout(wte, params["embed_dropout"], train)

        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * params["n_layer"]
        assert len(pasts) == params["n_layer"]
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, params=params, train=train)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', params=params)

        h_flat = tf.reshape(h, [batch*sequence, params["n_embd"]])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, params["n_vocab"]])
        results['logits'] = logits
        return results
