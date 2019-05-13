import tensorflow as tf

# Requires a valid encoder location
def gpt2_predict(predictions, params):
    from models.gpt2 import encoder

    enc = encoder.get_encoder(params["encoder_path"])

    with tf.gfile.Open(params["predict_path"], "a") as f:
        for i, p in enumerate(predictions):
            p = p["tokens"]
            text = enc.decode(p)
            f.write("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            f.write(text)
            f.write("\n" + "=" * 80 + "\n")