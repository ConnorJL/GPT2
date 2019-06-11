import random

import numpy as np

# Basic parameters of a 117M GPT2 network
gpt2_small = {
    "n_head": 12,
    "encoder_path": "gs://openwebtext/stuff/encoder",
    "n_vocab": 50257,
    "embed_dropout": 0.1,
    "lr": 0.00025,
    "warmup_steps": 2000,
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 1e-9,
    "opt_name": "adam",
    "weight_decay": 0.01,
    "train_batch_size": 32,
    "attn_dropout": 0.1,
    "train_steps": 10000,
    "eval_steps": 10,
    "max_steps": 500000,
    "data_path": "gs://connors-datasets/openwebtext/",
    "scale": 0.2886751345948129,
    "res_dropout": 0.1,
    "predict_batch_size": 1,
    "eval_batch_size": 32,
    "iterations": 500,
    "n_embd": 768,
    "input": "openwebtext",
    "model": "GPT2",
    "model_path": "gs://connors-models/GPT2-117M",
    "n_ctx": 1024,
    "predict_path": "logs/predictions.txt",
    "n_layer": 12
}

# Running parameters, each experiment needs a unique name, whether the TPU should be preemptible
# what type of TPU to use (GPUs TODO) and the actual model parameters
experiment_base = {
    "name": "gpt2_small",
    "preemptible": True,
    "accelerator_type": "v2-8",
    "model_params": gpt2_small,
}

# A class defining a hyperparameter to vary
class HyperParameter(object):
    def __init__(self, name, distribution, model=True, dtype=None, values=None):
        self.name = name # Name of the parameter to vary
        self.distribution = distribution # Which distribution should be used to generate the values, one of grid, sample, uniform or geometric
        self.values = values # Values to use by the distribution
        self.dtype = dtype # Whether to generate floats or ints, if necessary
        self.model = model # Whether the parameter belongs in the model_params or not

        if distribution == "grid" or distribution == "sample":
            assert type(values) == type([])
            self.values = values
            self.index = 0

        elif distribution == "uniform":
            assert type(values) == type([])
            assert len(values) == 2

        elif distribution == "geometric":
            assert type(values) == type([])
            assert len(values) == 2

    def get_value(self):
        # Simply iterate over a list of values
        if self.distribution == "grid":
            if self.index < len(self.values):
                val = self.values[self.index]
                self.index += 1
                return val
            else:
                raise RuntimeError("{} ran out of values!".format(self.name))

        # Sample randomly from a list of values
        elif self.distribution == "sample":
            return random.sample(self.values)

        # Sample from a uniform distribution
        elif self.distribution == "uniform":
            if self.dtype == "float":
                return random.uniform(self.values[0], self.values[1])
            else:
                return int(random.uniform(self.values[0], self.values[1]))

        # Sample from a "geometric" distribution
        # A sample is drawn from the uniform distribution from log(A) to log(B) and then expontiated to generate the value
        elif self.distribution == "geometric":
            if self.dtype == "float":
                return np.exp(np.random.uniform(np.log(self.values[0]), np.log(self.values[1])))
            else:
                return int(np.exp(np.random.uniform(np.log(self.values[0]), np.log(self.values[1]))))


# Given the base parameters of the model, list of parameter to vary and a number, generates number amount of experiments to run
def generate_experiments(base, parameters, number):
    experiments = []
    for i in range(number):
        ex = base.copy()
        ex["name"] = ex["name"] + "-" + str(i)
        ex["model_params"]["model_dir"] = ex["model_params"]["model_dir"] + "-" + str(i)

        for p in parameters:
            if p.model:
                ex["model_params"][p.name] = p.get_value()
            else:
                ex[p.name] = p.get_value()

        experiments.append(ex)

    return experiments

parameters = [
    HyperParameter("lr", "geometric", values=[1e-5, 1e-1], dtype="float"),
    HyperParameter("input", "sample", values=["openwebtext", "openwebtext_long", "openwebtext_longbiased"]),
    HyperParameter("n_layers", "uniform", values=[12, 24])
]

# This is what is exported to Overrunner to run
experiments = generate_experiments(experiment_base, parameters, 10)
