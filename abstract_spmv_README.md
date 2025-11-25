# Abstract SpMV Standalone Implementation

这是从 Ginkgo 的 `csr_kernels.template.cpp` 中提取的 `abstract_spmv` kernel 的独立实现。

## 文件说明

### 1. abstract_spmv_standalone.cu
包含完整的 `abstract_spmv` kernel 实现及其所有依赖项，特化为 `IndexType=int32, ValueType=double`。

**主要组件：**
- **数学工具函数**: `zero()`, `ceildivT()`, `min()`, `max()`
- **Atomic 操作**: `atomic_add()` for double
- **Accessor 系统**: 简化的 2D 和 1D accessor 用于访问矩阵和向量数据
- **Cooperative Groups**: 封装 CUDA cooperative groups
- **Segment Scan**: 用于 warp 内的分段扫描操作
- **SpMV Kernel 辅助函数**:
  - `find_next_row()`: 定位下一个非零元素所在的行
  - `warp_atomic_add()`: 使用 warp 内 reduction 的原子加法
  - `process_window()`: 处理一个 warp 的工作窗口
  - `get_warp_start_idx()`: 计算 warp 的起始索引
  - `spmv_kernel()`: 主要的 SpMV 计算逻辑
- **abstract_spmv kernels**: 两个版本
  - 不带 alpha 缩放的版本
  - 带 alpha 缩放的版本

### 2. abstract_spmv_test.cu
完整的测试示例，展示如何使用提取的 kernel。

**测试内容：**
- 创建一个 4×4 的 CSR 稀疏矩阵
- 执行矩阵-向量乘法 (y = A * x)
- 验证结果的正确性

**测试矩阵：**
```
[2  0  1  0]
[0  3  0  2]
[1  0  4  0]
[0  2  0  5]
```

**输入向量：** `[1, 2, 3, 4]^T`

**预期输出：** `[5, 14, 13, 24]^T`

## 编译和运行

### 前置要求
- NVIDIA CUDA Toolkit (建议 11.0 或更高版本)
- 支持 CUDA 的 GPU (计算能力 6.0 或更高)
- C++14 或更高版本的编译器

### 编译测试程序

```bash
# 使用 nvcc 编译
nvcc -std=c++14 -arch=sm_60 abstract_spmv_test.cu -o abstract_spmv_test

# 或者使用更高的计算能力
nvcc -std=c++14 -arch=sm_70 abstract_spmv_test.cu -o abstract_spmv_test
```

### 运行测试

```bash
./abstract_spmv_test
```

预期输出示例：
```
Testing standalone abstract_spmv implementation
================================================

Launching kernel with:
  Grid: (1, 1, 1)
  Block: (32, 4, 1)
  Number of warps: 1
  NNZ: 8

Results:
  c[0] = 5.000000 (expected: 5.000000) ✓
  c[1] = 14.000000 (expected: 14.000000) ✓
  c[2] = 13.000000 (expected: 13.000000) ✓
  c[3] = 24.000000 (expected: 24.000000) ✓

SUCCESS: All results match expected values!
```

## 集成到你的项目

### 方式 1: 直接包含源代码

将 `abstract_spmv_standalone.cu` 的内容复制到你的项目中。

### 方式 2: 编译为独立的编译单元

```bash
# 编译为目标文件
nvcc -std=c++14 -arch=sm_60 -dc abstract_spmv_standalone.cu -o abstract_spmv.o

# 在你的项目中链接
nvcc your_code.cu abstract_spmv.o -o your_program
```

### 使用示例

```cpp
// 1. 准备 CSR 矩阵数据
double* d_values;        // 非零元素值
int32* d_col_idxs;       // 列索引
int32* d_row_ptrs;       // 行指针
int32 num_rows, nnz;

// 2. 准备输入和输出向量
double* d_b;  // 输入向量
double* d_c;  // 输出向量（需要初始化为 0）

// 3. 计算 srow 数组
int32 nwarps = ceildivT<int32>(nnz, 32);
int32* d_srow;
cudaMalloc(&d_srow, nwarps * sizeof(int32));
compute_srow_kernel<<<blocks, threads>>>(num_rows, nwarps, d_row_ptrs, d_srow);

// 4. 创建 accessors 和 ranges
acc::simple_1d<double, int32> val_acc(d_values);
acc::simple_row_major_2d<double, int32> b_acc(num_cols, 1, d_b, 1);
acc::simple_row_major_2d<double, int32> c_acc(num_rows, 1, d_c, 1);

acc::range<acc::simple_1d<double, int32>> val_range(val_acc);
acc::range<acc::simple_row_major_2d<double, int32>> b_range(b_acc);
acc::range<acc::simple_row_major_2d<double, int32>> c_range(c_acc);

// 5. 启动 kernel
dim3 block_dim(config::warp_size, warps_in_block);
dim3 grid_dim(ceildivT<int32>(nwarps, warps_in_block), 1);

abstract_spmv<<<grid_dim, block_dim>>>(
    nwarps, num_rows, val_range, d_col_idxs, d_row_ptrs, d_srow,
    b_range, c_range);
```

## 技术细节

### Kernel 配置
- **Block size**: 128 threads (32 threads × 4 warps)
- **每个 warp**: 处理矩阵的一部分非零元素
- **工作分配**: 基于非零元素数量动态分配给各个 warp

### 性能特性
- 使用 warp-level primitives 进行高效的并行 reduction
- Segment scan 避免了不必要的 atomic 操作
- 针对不规则稀疏模式优化的负载均衡

### 限制
- 当前实现仅支持 `IndexType=int32` 和 `ValueType=double`
- 输出向量必须预先初始化为 0
- 适用于中等到大规模的稀疏矩阵

## 扩展

如果需要支持其他数据类型，可以：

1. **添加 float 支持**: 将所有 `double` 替换为模板参数
2. **添加 int64 索引**: 修改 `IndexType` 的定义
3. **支持复数**: 添加对 `thrust::complex` 的支持

## 依赖项说明

所有依赖项都已经内联到独立文件中：
- ✓ Accessor 系统（简化版）
- ✓ Cooperative groups 封装
- ✓ Atomic 操作
- ✓ Segment scan
- ✓ 数学工具函数

**无需额外的 Ginkgo 库依赖！**

## 许可证

本代码遵循 BSD-3-Clause 许可证，与原始 Ginkgo 项目保持一致。

## 参考

原始实现来自 Ginkgo 项目：
- 文件: `common/cuda_hip/matrix/csr_kernels.template.cpp`
- 函数: `abstract_spmv` (lines 251-288)

## 常见问题

**Q: 为什么输出结果不正确？**
A: 确保输出向量 `d_c` 在调用 kernel 前已经初始化为 0。

**Q: 如何处理多列的向量矩阵乘法？**
A: 修改 grid 的 y 维度：`dim3 grid_dim(ceildivT(nwarps, warps_in_block), num_cols);`

**Q: 性能不如预期？**
A:
- 检查矩阵大小是否足够大以充分利用 GPU
- 尝试调整 `warps_in_block` 的值
- 确保数据在设备内存中是连续的

**Q: 编译时出现 cooperative groups 相关错误？**
A: 确保使用 CUDA 9.0 或更高版本，并且指定了正确的计算能力。
