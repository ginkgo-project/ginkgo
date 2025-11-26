# Abstract SpMV Minimal - 项目概览

这是从 Ginkgo 的 `csr_kernels.template.cpp` 中提取的 `abstract_spmv` kernel 的完全独立、最小化实现。

## 📁 目录结构

```
abstract_spmv_minimal/
├── README.md                      # 完整使用文档
├── COMPILATION_FIXES.md           # 编译问题修复说明
├── PROJECT_OVERVIEW.md            # 项目概览（本文件）
│
├── abstract_spmv_standalone.cu    # 核心实现（~500行）
├── abstract_spmv_test.cu          # 完整测试示例（~600行）
├── test_minimal.cu                # 最小编译测试（~200行）
│
├── Makefile                       # 构建系统
├── compile_test.sh                # 自动化编译测试脚本
└── syntax_check.cpp               # C++语法检查工具
```

## 🎯 核心文件说明

### 1. abstract_spmv_standalone.cu
**独立的 SpMV kernel 实现**

包含内容：
- ✅ 完整的 `abstract_spmv` kernel（特化为 int32/double）
- ✅ 所有必需的辅助代码：
  - 数学工具函数（zero, ceildivT, min, max）
  - Atomic 操作
  - 简化的 Accessor 系统
  - Cooperative groups 封装
  - Segment scan 实现
  - SpMV 辅助函数

**无需任何 Ginkgo 库依赖！**

### 2. abstract_spmv_test.cu
**完整的测试程序**

特点：
- 包含完整的 kernel 代码
- 4×4 CSR 矩阵测试用例
- 自动验证结果
- 可独立编译运行

测试矩阵：
```
[2  0  1  0]     [1]     [5]
[0  3  0  2]  ×  [2]  =  [14]
[1  0  4  0]     [3]     [13]
[0  2  0  5]     [4]     [24]
```

### 3. test_minimal.cu
**快速编译验证**

- 最小化的测试代码
- 用于快速验证编译环境
- 编译时间短

## 🚀 快速开始

### 方式 1: 自动化测试（推荐）

```bash
cd abstract_spmv_minimal
./compile_test.sh
```

此脚本会自动：
1. ✅ 编译最小测试
2. ✅ 编译独立实现
3. ✅ 编译并运行完整测试
4. ✅ 验证结果

### 方式 2: 使用 Makefile

```bash
cd abstract_spmv_minimal
make          # 编译测试
make run      # 编译并运行
make clean    # 清理
```

### 方式 3: 手动编译

```bash
cd abstract_spmv_minimal

# 最小测试
nvcc -std=c++14 -arch=sm_70 test_minimal.cu -o test_minimal
./test_minimal

# 完整测试
nvcc -std=c++14 -arch=sm_70 abstract_spmv_test.cu -o abstract_spmv_test
./abstract_spmv_test
```

## 📊 技术规格

### 支持的配置
- **IndexType**: int32（32位整数索引）
- **ValueType**: double（双精度浮点）
- **矩阵格式**: CSR (Compressed Sparse Row)

### 系统要求
- CUDA Toolkit 11.0+
- 计算能力 6.0+ (sm_60+)
- C++14 编译器

### Kernel 配置
- **Block size**: 128 threads (32×4)
- **Warp size**: 32
- **Warps per block**: 4

## 🔧 集成到你的项目

### 方法 1: 直接包含源代码
```cpp
#include "abstract_spmv_standalone.cu"
// 使用 abstract_spmv kernel
```

### 方法 2: 编译为目标文件
```bash
nvcc -std=c++14 -arch=sm_70 -dc abstract_spmv_standalone.cu -o abstract_spmv.o
# 链接到你的项目
nvcc your_code.cu abstract_spmv.o -o your_program
```

### 方法 3: 复制必要代码
从 `abstract_spmv_standalone.cu` 中提取需要的部分集成到你的代码库。

## 📖 文档

- **README.md** - 完整使用指南
  - 编译和运行说明
  - API 文档
  - 使用示例
  - 常见问题

- **COMPILATION_FIXES.md** - 编译问题修复
  - 问题诊断
  - 解决方案
  - 技术细节

- **PROJECT_OVERVIEW.md** - 项目概览（本文件）
  - 文件结构
  - 快速开始
  - 技术规格

## 🎓 学习资源

### 理解代码结构
1. 从 `test_minimal.cu` 开始 - 了解基本结构
2. 阅读 `abstract_spmv_standalone.cu` - 理解 kernel 实现
3. 研究 `abstract_spmv_test.cu` - 学习如何使用

### 关键概念
- **CSR 格式**: 压缩稀疏行格式
- **Accessor 系统**: 抽象的内存访问接口
- **Segment scan**: Warp 内的分段扫描
- **Warp 分配**: 动态负载均衡

## 📝 修改历史

### v1.0 - 初始提取
- 从 Ginkgo csr_kernels.template.cpp 提取
- 特化为 int32/double
- 包含所有依赖项

### v1.1 - 编译修复
- 修复 decltype 编译错误
- 添加 std::declval 支持
- 添加测试工具

### v1.2 - 目录重组
- 移动所有文件到专门目录
- 改进文档结构
- 添加项目概览

## 🔗 相关链接

- **原始 Ginkgo 项目**: https://github.com/ginkgo-project/ginkgo
- **CSR 格式说明**: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
- **CUDA 编程指南**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## 📄 许可证

BSD-3-Clause License（与 Ginkgo 项目保持一致）

---

**准备好开始了吗？运行：**
```bash
cd abstract_spmv_minimal && ./compile_test.sh
```
