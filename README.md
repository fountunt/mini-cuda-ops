# mini-cuda-ops

手写CUDA算子库，学习GPU编程模型与性能优化

每个算子包含三部分：**手写kernel** · **PyTorch `torch.allclose` 校验** · **benchmark与瓶颈分析**

## 环境

GPU:RTX 5060 Laptop(8GB,Blackwell,**`sm_120`**)
CUDA 12.8
系统:WSL2 + Ubuntu 24.04
PyTorch 2.x

## 编译与运行

```bash
nvcc -arch=sm_120 src/vector_add.cu -o vector_add
./vector_add
```

## 算子进度

| 算子 | kernel | Pytorch校验 | benchmark |
|---|---|---|---|
| vector_add | ⬜| ⬜| ⬜|
| reduction | ⬜| ⬜| ⬜|
| softmax | ⬜| ⬜| ⬜|
| layernorm | ⬜| ⬜| ⬜|
| GEMM(naive) | ⬜| ⬜| ⬜|
| INT8量化GEMM | ⬜| ⬜| ⬜|
