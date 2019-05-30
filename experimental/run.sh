export TPU_NAME=$1
export MODEL=$2
export SCRIPT_NAME=main.py

python3 $SCRIPT_NAME \
        --tpu $TPU_NAME \
        --model $MODEL