import os
import sys
import requests
from tqdm import tqdm

if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 117M')
    sys.exit(1)

model = sys.argv[1]

if model not in ["117M", "PrettyBig", "encoder"]:
    print("Unknown model! Currently available models: 117M, SortaBig")
    sys.exit(1)


if not os.path.exists(model):
    os.makedirs(model)
if not os.path.exists("encoder"):
    os.makedirs("encoder")

for filename in ['encoder.json', 'vocab.bpe']:

    r = requests.get("https://storage.googleapis.com/connors-models/public/encoder/" + filename, stream=True)

    with open(os.path.join("encoder", filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

if model == "encoder":
    sys.exit()

with open(os.path.join(model, "checkpoint"), "w") as f:
        f.write("model_checkpoint_path: \"model.ckpt\"\n")

for filename in ['model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta']:

    r = requests.get("https://storage.googleapis.com/connors-models/public/" + model + "/" + filename, stream=True)

    with open(os.path.join(model, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
