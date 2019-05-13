from functools import partial

import numpy as np
import tensorflow as tf

from optimizers import create_train_op
from metric_fns import *

def gpt2_model(features, labels, mode, params):
    from models.gpt2 import gpt2

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        output = gpt2.model(X=features, params=params,
                            labels=labels,
                            past=None, reuse=tf.AUTO_REUSE,
                            eval=mode==tf.estimator.ModeKeys.EVAL,
                            train=mode==tf.estimator.ModeKeys.TRAIN)
        loss = output["loss"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_train_op(loss, params)
    
        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == tf.estimator.ModeKeys.EVAL:
        from metric_fns import perplexity_metric

        if params["use_tpu"]:
            # Metric inputs are transferred to CPU and must preserve batch dimension
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, 
                loss=loss, eval_metrics=(perplexity_metric, {"loss": output["raw_loss"]}))
        else:
            return tf.estimator.EstimatorSpec(mode=mode, 
                loss=loss, eval_metrics=perplexity_metric(output["raw_loss"]))


    if mode == tf.estimator.ModeKeys.PREDICT:
        from models.gpt2 import sample

        output = sample.sample_sequence(
            params=params, length=params["n_ctx"],
            context=features,
            batch_size=params["batch_size"],
            temperature=1.0, top_k=0
        )
        
        predictions = {
            "tokens": output
        }

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def ResNeXt_Spines_model(features, labels, mode, params):
    # Features = context, labels = real text
    from models.ResNeXt_Spines import ResNext_Spines

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        output = ResNext_Spines.model(X=features, params=params,
                            labels=labels,
                            reuse=tf.AUTO_REUSE,
                            scope="ResNeXt_Spines",
                            eval=mode==tf.estimator.ModeKeys.EVAL,
                            train=mode==tf.estimator.ModeKeys.TRAIN)
        loss = output["loss"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_train_op(loss, params)
    
        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == tf.estimator.ModeKeys.EVAL:

        if params["use_tpu"]:
            # Metric inputs are transferred to CPU and must preserve batch dimension
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, 
                loss=loss, eval_metrics=(mean_squared_error, {"loss": output["raw_loss"]}))
        else:
            return tf.estimator.EstimatorSpec(mode=mode, 
                loss=loss, eval_metrics=mean_squared_error(output["raw_loss"]))


    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

