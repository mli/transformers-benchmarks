# Transformers Benchmarks

We benchmark real [TeraFLOPS](https://en.wikipedia.org/wiki/FLOPS) that training Transformer models can achieve on various GPUs, including single GPU, multi-GPUs, and multi-machines. It helps you to estimate how many machine times you need to train your large-scale Transformer models.

The real performance depends on multiple factors, including your hardware, cooling, CUDA version, transformer models, hyper-parameters such as batch sizes, and implementations. We list the numbers we got on both personal PC and cloud instances. We also provide Jupyter notebooks for you to benchmark on your machines and workloads:

- [Understanding Transformer layer performance](micro_bench.ipynb)
- [Training BERT and GPT-2 with (multi-)GPUs](transformers.ipynb)

## Micro-Benchmarking Summary

Measure the TFLOPS for various micro-benchmarkings. Results are from running [micro_bench.ipynb](micro_bench.ipynb).

|                                        | A100      |  A6000   | V100      | 3090 Ti  | 4090 |
| -------------------------------------- | :-------: | :------: | :-------: | :------: | :---: |
| Theory TF32(FP32) / FP16               | 156 / 312 | 75 / 150 | 16 / 125  | 80 / 160 |
| Memory (GB) / Bandwidth (GB/s)         | 80 / 2039 | 48 / 768 | 32 / 900  | 24 / 1008 | 24 / 1008 |
| Approximate Price $                    |  16,000   |  4,000   |   3,500   |  1,500   | 2,200 |
| Matrix Multiplication FP32 / FP16      | 116 / 230 | 60 / 95  |  14 / 95  | 42 / 81  | 86 / 172 |
| Vector Multiplication                  |   0.202   |  0.082   |   0.098   |  0.107   |  0.117 |
| Bert Layer Forward / Forward+Backward  | 110 / 136 | 60 / 70  |  53 / 64  | 56 / 62  | 99 / 109 |
| GPT-2 Layer Forward / Forward+Backward |  45 / 53  | 35 / 38  |  32 / 36  | 37 / 39  | 48 / 54 |
| T5 Encoder Forward / Forward+Backward  |  44 / 56  | 34 / 41  |  31 / 38  | 36 / 41  | 47 / 55 |
| T5 Decoder Forward / Forward+Backward  |  38 / 47  | 28 / 34  |  26 / 32  | 30 / 36  | 38 / 45 |



## Set Up

You need a CUDA-enabled pytorch to run workloads. We recommend you to use the latest version CUDA and pytorch for better performance. One easy way is using [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Once installed, you can find latest tag of the [pytorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), for exmaple, `22.07-py3`, then run

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 -v ~:/workspace \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	nvcr.io/nvidia/pytorch:22.07-py3
```

After the docker is running, execute  `jupyter notebook` in the docker's shell to open this notebook.
