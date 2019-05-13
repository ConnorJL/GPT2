import os

import tensorflow as tf

# Expects .tfrecords files as produced by the script in datasets in a google storage bucket

def openwebtext_train(params, batch=True):
    numbers = [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, # 32, (32 is empty)
                33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 63, 64,
                65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92]
    files = [os.path.join(params["data_path"], "openwebtext-newspaper_{}.tfrecords".format(str(i))) for i in numbers]

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=42, batch=batch)

def openwebtext_eval(params, batch=True):
    numbers = [1, 2, 11, 12, 16, 34, 36, 58, 60, 61, 62, 67, 83]
    files = [os.path.join(params["data_path"], "openwebtext-newspaper_{}.tfrecords".format(str(i))) for i in numbers]

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=42, batch=batch)

def openwebtext_long_train(params, batch=True):
    numbers = [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 63, 64,
                65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92]
    files = [os.path.join(params["data_path"], "openwebtext-newspaper-long_{}.tfrecords".format(str(i))) for i in numbers]

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=2, batch=batch)

def openwebtext_long_eval(params, batch=True):
    numbers = [1, 2, 11, 12, 16, 34, 36, 58, 60, 61, 62, 67, 83]
    files = [os.path.join(params["data_path"], "openwebtext-newspaper-long_{}.tfrecords".format(str(i))) for i in numbers]

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=2, batch=batch)

def openwebtext_longbiased_train(params):
    datasets = [openwebtext_train(params, batch=False), openwebtext_long_train(params, batch=False)]
    weights = [0.3, 0.7]

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["batch_size"], drop_remainder=True).prefetch(params["iterations"])

    return dataset

def openwebtext_longbiased_eval(params):
    datasets = [openwebtext_eval(params, batch=False), openwebtext_long_eval(params, batch=False)]
    weights = [0.3, 0.7]

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["batch_size"], drop_remainder=True).prefetch(params["iterations"])

    return dataset

def shakespeare_test(params):
    files = ["gs://openwebtext/stuff/data.tfrecords"]

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=1)

def generic_text(params):
    # params["datasets"] = [([files], weight)]
    datasets = [bpe_text(params["batch_size"], dataset[0], amount=params["n_ctx"], iterations=params["iterations"], stitch=params["stitch"], batch=False)
                for dataset in params["dataset"]]
    weights = [dataset[1] for dataset in params["dataset"]]
    
    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    return dataset

def bpe_text(batch_size, files, iterations, stitch, amount=1024, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=True))

    def _parse_function(example_proto):
        features = {
            "hash": tf.VarLenFeature(tf.string),
            "text": tf.VarLenFeature(tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["text"], parsed_features["text"].dense_shape[0]

    dataset = dataset.map(_parse_function, num_parallel_calls=1).shuffle(1000 * stitch)

    def _stitch_text(x, y): # Relies on the fact that stitch * min(characters_in_text) >= amount
        x = tf.sparse.to_dense(x)

        def _get_x(i):
            return tf.gather(x[i], tf.range(y[i]))

        out = _get_x(0)
        for i in range(1, stitch):
            out = tf.concat([out, [50256], _get_x(i)], axis=0) # text1<|endoftext|>text2
        
        return out

    # Hack-y way to stitch together multiple texts
    dataset = dataset.batch(stitch, drop_remainder=True).map(_stitch_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _sample_text(x):
        s = tf.size(x)
        r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int32)
        r1 = tf.range(r, r+amount)
        r2 = tf.range(r+1, (r+1)+amount)
        r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
        r2 = tf.reshape(r2, [amount])
        vals1 = tf.gather(x, r1)
        vals2 = tf.gather(x, r2)
        return vals1, vals2

    if batch:
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=_sample_text, batch_size=batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
        dataset = dataset.repeat().prefetch(iterations)

    else:
        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()

    return dataset

def gpt2_pred_input(params):
    t = tf.fill([params["batch_size"], 1], 15496) # "Hello"
    dataset = tf.data.Dataset.from_tensors(t) 
    return dataset