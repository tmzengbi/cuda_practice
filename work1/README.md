使用下面一种或多种优化方法完成 CUDA 的矩阵向量乘法 $y = A \times x$, 其中 $A$ 是 $2^14 \times 2^14$ 的方阵, $x$ 是 $2^14$ 维向量。假设矩阵 $A$ 的元素为 $a_{ij}=i - 0.1 \times j + 1$, 向量 $x$ 的元素为 $b_i = \log\sqrt{i\times i - i + 2}$ 

- 使用 global memory
    本来想在 cpu 上提前计算好 $x$ 再传输到 gpu 上，考虑到传输延迟，决定直接在 gpu 上进行计算。
    通过查询文档得到了 gpu 上单精度的数学库 https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.（单精度），html#group__CUDA__MATH__SINGLE  https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE（双精度）
    其中单精度耗费时间 679ms，双精度耗费时间 3828ms

    

- 使用合并访存
- 使用 constant memory 存放向量
- 使用 shared memory 存放向量和矩阵
- 使用 warp 直接访问寄存器
- 使用 cublasSgemv