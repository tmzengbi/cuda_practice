## 任务
    使用下面一种或多种优化方法完成 CUDA 的矩阵向量乘法 $y = A \times x$, 其中 $A$ 是 $2^14 \times 2^14$ 的方阵, $x$ 是 $2^14$ 维向量。假设矩阵 $A$ 的元素为 $a_{ij}=i - 0.1 \times j + 1$, 向量 $x$ 的元素为 $b_i = \log\sqrt{i\times i - i + 2}$ 

## 环境
GeForce rtx 3060 laptop 115w，cuda on wsl,**其中，所有的核函数都使用 <<<2,16>>> 并行计算**

## 实现

### 使用 global memory

    本来想在 cpu 上提前计算好 $x$ 再传输到 gpu 上，考虑到传输延迟，决定直接在 gpu 上进行计算。
    通过查询文档得到了 gpu 上单精度的数学库 https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE. （单精度，见 main10.cu）  
    https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE （双精度，见 main11.cu）
    
    其中单精度耗费时间 1617ms，双精度耗费时间 7729ms

    但是实际上传输延迟似乎没有想象中那么大，使用单精度，如果提前在 host 中计算好 $x$，只需要 1122ms 就可以完成。

### 使用合并访存

    找了好久什么是合并访存无果，看 wu-kan 博客才了解他说的合并访存就是 cache 友好的访存方式 :(，这里无需实现。

### 使用 constant memory 存放向量

    cuda 提供了 64 kb 的 constant memory，恰好是 $2^{14}$ 个 float 变量的大小，可以考虑将所有 $x$ 都存入 constant memory 中，只需要 847ms 计算完毕。

### 使用 shared memory 存放向量和矩阵

    本地环境电脑使用了 rtx3060 laptop，经查询 https://developer.nvidia.com/cuda-gpus ，computing capacity 为 8.6，由官方文档的说明 https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities ，该 gpu 每个 block 可以分配最大 99kb 的shared memory，但是实际上却运行失败，经检验，还是只能拥有最大 48kb 的 shared memory。查询相关问题，找到解决方法 https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda

    除此之外，使用大于 48kb 的 shared memory 必须使用 dynamic allocate。使用 shared memory 的计算时间 916ms

### 使用 warp 直接访问寄存器

### 使用 cublasSgemv


## 遇到的 bug
1. 使用 cuda 提供的数学函数 api log 的时候，报 compile error，后发现是因为 log 函数里面数据类型是 int，导致编译器识别为标准数学库函数，host 函数无法在 device 上面使用。