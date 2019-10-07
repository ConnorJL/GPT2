import glob
import os
import time
from multiprocessing import Pool

import ftfy
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from absl import flags

import encoder

FLAGS = flags.FLAGS

base_dir = "/home/connor/2/newspaper" # Path to where your .txt files are located
files_per = 175000 # 175000 ~ 200-300MB
name = "openwebtext-newspaper" # Name of output files will be name_i.tfrecords where i is the number of the file
output_dir = "/home/connor/out"
log_dir = "logs"

processes = 64 # Number of encoding processes to run
encoder_path = "gs://openwebtext/stuff/encoder" # Path to encoder files
minimum_size = 25

flags.DEFINE_string(
    "base_dir",
    default="/home/connor/2/newspaper",
    help="Path to where your .txt files are located.")


flags.DEFINE_string(
    "output_dir",
    default="/home/connor/out",
    )
flags.DEFINE_string(
    "log_dir",
    default="logs",
    )
flags.DEFINE_string(
    "encoder_path",
    default="gs://openwebtext/stuff/encoder" ,
    help="Path to encoder files")


flags.DEFINE_string(
    "name",
    default="openwebtext-newspaper",
    help="Name of output files will be name_i.tfrecords where i is the number of the file")


flags.DEFINE_integer("processes",
                     default=64,
                    help="Number of encoding processes to run")
flags.DEFINE_integer("minimum_size",
                     default=25,
                     help="minimum text size"
                    )
flags.DEFINE_integer("files_per",
                     default=1500,
                    help="file chunk size")


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Divides a list into chunks
def chunks(l, n):
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

enc = encoder.get_encoder(encoder_path)

file_chunks = chunks(files, files_per)

print("Got {} files, divided into {} chunks.".format(str(len(files)), str(len(file_chunks))))

def create_file(args):
    i, chunk = args
    s = name + "_" + str(i) + ".tfrecords"
    if os.path.exists(os.path.join(log_dir, s)): # Hack-y, if file of same name is in log dir, sign that the file is complete, so skip
        return
    if os.path.exists(os.path.join(output_dir, s)): # Unfinished file, remove
        os.remove(os.path.join(output_dir, s))

    with tf.python_io.TFRecordWriter(os.path.join(output_dir, s)) as writer:
        good_files = 0
        current = None
        for fn in chunk:
            with tf.gfile.Open(fn, "r") as f:
                d = f.read()
            d = ftfy.fix_text(d, normalization='NFKC')
            data = np.array(enc.encode(d), np.int32)
            if data.shape[0] < minimum_size or (data == 0).all(): # If text is shorter than 25 tokens, or all tokens are 0, ignore
                continue
            hash = fn.split("/")[-1].split(".")[0]
            feature = {
                "hash": _bytes_feature(hash.encode()),
                "text": _int64_feature(data)
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            good_files += 1
    # File complete
    with open(os.path.join(log_dir, s), "w") as f: # Create mark that file is finished in logdir
        f.write("{} / {}".format(str(good_files), str(len(chunk))))
    with open(os.path.join(log_dir, "good_files.log"), "a") as f:
        f.write("{}: {} / {}".format(str(i), str(good_files), str(len(chunk))))

    return good_files


def main(argv  ):
    if FLAGS.debug:
        print('non-flag arguments:', argv)
    base_dir = FLAGS.base_dir ,# Path to where your .txt files are located
    files_per =FLAGS.files_per ,# 175000 ~ 200-300MB
    name = FLAGS.name, # Name of output files will be name_i.tfrecords where i is the number of the file
    output_dir = FLAGS.output_dir,
    log_dir = FLAGS.log_dir,
    processes = FLAGS.processes, # Number of encoding processes to run
    encoder_path =FLAGS.encoder_path ,# Path to encoder files
    minimum_size = FLAGS.minimum_size
    
    files = glob.glob(os.path.join(base_dir, "**/*.txt"))

    start = time.time()
    pool = Pool(processes=processes)
    good = 0
    for g in tqdm(pool.imap(create_file, enumerate(file_chunks)), total=len(file_chunks)):
        good += g

    end = time.time()

    print("Done! In {:.2f}s, {} / {} good files.".format(end-start, str(good), str(len(files))))

if __name__ == '__main__':
  app.run(main)
