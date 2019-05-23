# GPT2
**This is not the official GPT2 implentation!**

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/) that supports both GPUs and TPUs. The dataset scripts are hack-y and will probably need to be adapted to your needs. 
## Requirements
For GPUs:

`pip3 install tensorflow-gpu regex`

For TPUs:

`pip3 install tensorflow regex`

`pip3 install --upgrade google-api-python-client`

`pip3 install --upgrade oauth2client`

## Training
To train a model, define its parameters in a .json file (see examples) and then simply call

`python3 main.py --model Your-Model.json --tpu=Your-TPU-Name`

Using a TPU is optional, it runs fine on GPUs without modification. (Note: Evaluation doesn't work on TPU pods and must be commented out) 

This assumes you have a version of the openwebtext corpus stored in an accessible location, if you don't see below how to generate your own version.

## Generating Text
To predict you can either pass the prompt directly in the command line, or have it read from a file. (This is useful for prompts that include new lines) Text is output to the console and the file specified in the "predict_path" parameter.

From command line:

`python3 main.py --model Your-Model.json --predict_text "Hello there! My name is"`

From file:

`python3 main.py --model Your-Model.json --predict_file input.txt`

Prediction on TPUs is not supported.

## Generating the Dataset
GPT2 is trained on the webtext corpus, which is basically all websites linked to from reddit with at least 3 Karma. Since the database is huge and contains a lot of copyrighted material, I can't provide a download here. Instead I'll describe how I got it. Be aware it cost me around ~500€ in cloud compute resources to donwload and process the whole thing, but I'm not claiming I was optimally efficient. 
1. Use the download script from [here](https://github.com/jcpeterson/openwebtext) to download the archives (I used the prefilteres URLs file)
2. Use *datasets/extract_text.py* and *datasets/run_newspaper_extract.py*  to extract the text. 
3. Once you have the raw .txt files use *datasets/create_tfrecords.py* to encode them into correct .tfrecords files.
4. Place the .tfrecords files into a Google Storage bucket. (This is mandatory if you're using TPUs)
5. Change the "data_path" parameter to point to where your files are located and, if necessary, adapt the functions in *inputs.py* to open the correct filenames, in case you changed them.


## Explanation of Parameters
The way the code is setup, you pass all the model parameters in a .json file. Note that any paths also support Google Storage paths.

* **model**: A string that refers to which model to use. This should always just be "GPT2"
* **model_dir**: Where to save and load checkpoints from
* **n_ctx**: Number of tokens the model looks at