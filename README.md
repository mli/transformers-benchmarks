# Transformers Benchmarks

We benchmark real [TeraFLOPS](https://en.wikipedia.org/wiki/FLOPS) that training Transformer models can achieve on various GPUs, including single GPU, multi-GPUs, and multi-machines. It helps you to estimate how many machine times you need to train your large-scale Transformer models. 

The real performance depends on multiple factors, including your hardware, cooling, CUDA version, transformer models, hyper-parameters such as batch sizes, and implementations. We list the numbers we got on both personal PC and cloud instances. We also provide Jupyter notebooks for you to benchmark on your machines and workloads.

|        | MatMul | BERT-large | GPT2-medium |
| ------ | ------ | ---------- | ----------- |
| V100   | 96     | 41.2       | 18          |
| 3090Ti | 81     | 48.9       | 21.1        |
|        |        |            |             |

## Setup

You need a CUDA-enabled pytorch to run workloads. We recommend you to use the latest version CUDA and pytorch for better performance. One easy way is using  [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Once installed, you can find latest tag of the [pytorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), for exmaple, `22.07-py3`, then run 

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 -v ~/transformers-benchmarks:/workspace --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:22.07-py3
```

After the docker is running, execute  `jupyter notebook` in the docker's shell to open this notebook.



## Bert-large

```
task = "mrpc"
model = "bert-large-uncased"
batch_size = 96
seq_len = 128
fp16 = True # default: False
optim = "adamw_apex_fused"  # default: adamw_hf
gradient_checkpointing = False # default: False
gradient_accumulation_steps = 4 # default: 1
```



