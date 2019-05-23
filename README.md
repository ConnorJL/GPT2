# GPT2
**This is not the official GPT2 implementation!**

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/) that supports both GPUs and TPUs. The dataset scripts are a bit hack-y and will probably need to be adapted to your needs. 
## Requirements
For GPUs:

`pip3 install tensorflow-gpu regex`

For TPUs:

`pip3 install tensorflow regex google-api-python-client oauth2client`

For downloading the models:

`pip3 install requests tqdm`

For generating the dataset (in addition to Tensorflow):

`pip3 install ftfy tqdm newspaper3k`

## Downloading Pretrained Models
If you want to use my models, I currently have "117M" and "PrettyBig" to offer. 117M was trained on a single v2 TPU for a week (probably less than the original OpenAI model), PrettyBig is slightly bigger than 345M and was trained on a v2-256 pod for a week.

`python3 download_model.py PrettyBig`

This will create two directories, one named as the model and another named "encoder". Change the "model_dir" and "encoder_path" parameters in the .json corresponding to your model to point to these paths, respectively.

If you only want the encoder, use:

`python3 download_model.py encoder`

## Generating Text
To predict you can either pass the prompt directly in the command line, or have it read from a file. (This is useful for prompts that include new lines) Text is output to the console and the file specified in the "predict_path" parameter. You need a model checkpoint and a copy of the BPE encoder at an accessible location for this to work. (Change the "model_dir" and "encoder_path" parameters in the .json)

From command line:

`python3 main.py --model Your-Model.json --predict_text "Hello there! My name is"`

From file:

`python3 main.py --model Your-Model.json --predict_file input.txt`

Prediction on TPUs is not supported.


## Training
To train a model, define its parameters in a .json file (see examples) and then simply call

`python3 main.py --model Your-Model.json [--tpu Your-TPU-Name]`

Using a TPU is optional, it runs fine on GPUs without modification. (Note: Evaluation doesn't work on TPU pods and must be commented out) 

This assumes you have a version of the openwebtext corpus stored in an accessible location. If you don't, see below how to generate your own version.



## Generating the Dataset
GPT2 is trained on the webtext corpus, which is basically all websites linked to from reddit with at least 3 Karma. Since the database is huge and contains a lot of copyrighted material, I can't provide a download here. Instead I'll describe how I got it. Be aware it cost me around ~500€ in cloud compute resources to donwload and process the whole thing, but I'm not claiming I was optimally efficient. 
1. Use the download script from [here](https://github.com/jcpeterson/openwebtext) to download the archives (I used the prefiltered URLs file)
2. Use *datasets/run_newspaper_extract.py* to extract the text
3. Once you have the raw .txt files use *datasets/create_tfrecords.py* to encode them into .tfrecords files (Requires a copy of the encoder, see Downloading Pretrained Models)
4. Place the .tfrecords files into an accessible folder or Google Storage bucket (Placing in a Google Storage bucket is mandatory if you're using TPUs)
5. Change the "data_path" parameter in your .json to point to where your .tfrecords files are located and, if necessary, adapt the functions in *inputs.py* to open the correct filenames, in case you changed them


## Explanation of Parameters
Because passing two dozen parameters over the command line would be tedious, you pass all the model parameters in a .json file. Note that any paths also support Google Storage paths and *must* be gs:// paths if you're running on TPUs.

Values you'll definitely want to change:
* **model_path**: Where to save and load checkpoints from
* **data_path**: Where your .tfrecords files are located
* **encoder_path**: Path to the BPE encoder files. To get this, use the download_model.py script to download any model (or just the encoder). You will get a folder called "encoder". This is what you want this to point to (only required for prediction)

Values you'll probably want to change:
* **train_batch_size**: Batch size during training phase
* **eval_batch_size**: Batch size during evaluation
* **predict_batch_size**: Batch size during prediction
* **predict_path**: Where to save predictions (point this to a text file to append to)

Model parameters:
* **model**: A string that refers to which model to use. This should always just be "GPT2" (no other models are implemented here)
* **n_ctx**: Number of tokens the model looks at (default: 1024)
* **n_vocab**: Size of vocabulary (default: 50257)
* **n_embd**: Dimension of embedding layers
* **n_layer**: Number of layers in the model
* **n_head**: Number of attention heads (default: n_embd / 64)
* **scale**: Factor by which to scale initializations of weights (default: 1/sqrt(n_layer))

Training parameters:
* **input**: Which input function to use (default: "openwebtext")
* **lr**: Learning rate (default: 0.00025)
* **warmup_steps**: Number of warmup steps. If this is set, a linear warmup + cosine decay schedule is used (default: 2000)
* **opt_name**: Name of optimizer, currently only "adamW" is implemented (default: "adamW")
* **beta1**: Adam beta1 parameter (default: 0.9)
* **beta2**: Adam beta2 parameter (default: 0.98)
* **epsilon**: Adam epsilon parameter (default: 1e-9)
* **weight_decay**: Weight decay parameter (default: 0.01)
* **train_steps**: Number of training steps to take between evaluations
* **eval_steps**: Number of steps per evaluation
* **max_steps**: The maximum number of training steps (important for declining lr)
* **iterations**: Number of iterations to perform on TPUs (Only required for TPUs) (Default: 100)
* **embed_dropout**: Dropout chance on the word embedding (default: 0.1)
* **attn_dropout**: Dropout chance on attention layers (default: 0.1)
* **res_dropout**: Dropout chance on residual connections (default: 0.1)