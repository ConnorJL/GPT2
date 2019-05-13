import argparse
import json
import logging
from pathlib import Path
import sys
import time

import tensorflow as tf

from model_fns import *
from inputs import *
from predict_fns import *


models = {
    "GPT2": (gpt2_model, gpt2_predict)
}

inputs = {
    "openwebtext": (openwebtext_train, openwebtext_eval, gpt2_pred_input), # Standard OpenWebtext inout
    "shakespeare_test": (shakespeare_test, shakespeare_test, gpt2_pred_input), # Test dataset
    "openwebtext_longbiased": (openwebtext_longbiased_train, openwebtext_longbiased_eval, gpt2_pred_input), # OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": (openwebtext_long_train, openwebtext_long_eval, gpt2_pred_input) # Openwebtext that only shows long examples
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str) # Name of model, reads parameters from corresponding .json
    parser.add_argument("--predict", dest='predict', action='store_true') # If present, simply predicts rather than trains
    parser.set_defaults(predict=False)
    args = parser.parse_args()

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('logs/{}.log'.format(args.model)),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers

    # Read params of model
    with open(args.model + ".json", "r") as f:
        params = json.load(f)

    if not args.tpu is None:
        params["use_tpu"] = True
    else:
        params["use_tpu"] = False

    logger.info(params)

    model_fn = models[params["model"]][0]
    predict_fn = models[params["model"]][1]
    input_fn = inputs[params["input"]]

    if params["use_tpu"] and not args.predict:
        # Resolve TPU cluster and runconfig
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=params["model_dir"],
            cluster=tpu_cluster_resolver,
            save_checkpoints_secs=60*60*4,
            session_config=tf.ConfigProto(
                # allow_soft_placement=True, 
                # log_device_placement=True
                ),
                tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=params["iterations"])
        )

        # Set up network
        network = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                use_tpu=True,
                train_batch_size=params["train_batch_size"], # These are the global sizes, must be divisible by replicas
                eval_batch_size=params["eval_batch_size"],
                predict_batch_size=params["predict_batch_size"],
                config=run_config,
                params=params)

    else:
        # Non TPU setup
        params["batch_size"] = params["train_batch_size"]
        run_config = tf.estimator.RunConfig(
            model_dir=params["model_dir"],
            session_config=tf.ConfigProto(
                # log_device_placement=True,
                # allow_soft_placement=True
            ),
        )

        network = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params)

    if args.predict:
        logger.info("Generating predictions...")
        predictions = network.predict(input_fn=input_fn[2])
        predict_fn(predictions, params)
        sys.exit()

    while True:
        start = time.time()

        network.train(
                input_fn=input_fn[0],
                steps=params["train_steps"])

        end = time.time()
        logger.info("\nTrain loop took {:.2f}s\n".format(end-start))

        eval_result = network.evaluate(
           input_fn=input_fn[1],
           steps=params["eval_steps"])
        
        logger.info("\nEval Results: {}\n".format(str(eval_result)))

        if network.get_variable_value("global_step") > params["max_steps"]:
            logger.info("Done!")
            break
