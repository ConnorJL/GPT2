# GPT2
**This is not the official GPT2 implentation! Documentation TODO**

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/) that supports TPUs. The dataset scripts are hack-y and will probably need to be adapted to your needs. 

To train a model, define its parameters in a .json file (see examples) and then simply call

`python3 main.py --model=Your-Model --tpu=Your-TPU-Name`

Using a TPU is optional, it runs fine on GPUs without modification. (Note that evaluation doesn't work on TPU pods and must be commented out) To predict (currently only supporting a starting token of "Hello", TODO):

`python3 main.py --model=Your-Model --predict`

Prediction on TPUs is not supported.