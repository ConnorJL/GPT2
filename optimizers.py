import numpy as np
import tensorflow as tf


def create_train_op(loss, params):
    lr = params["lr"]
    if "warmup_steps" in params.keys():
        lr = cosine_decay_with_warmup(tf.train.get_global_step(), lr, params["max_steps"], warmup_steps=params["warmup_steps"])

    if params["opt_name"] == "adamW":
        optimizer = tf.contrib.opt.AdamWOptimizer(
            learning_rate=lr,
            weight_decay=lr*params["weight_decay"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            epsilon=params["epsilon"])
    else:
        raise ValueError("Unknown optimizer type!")
        
    if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # To update batchnorm, if present
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op = tf.group([train_op, update_ops])

    return train_op



def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                        'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                                learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                        'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')