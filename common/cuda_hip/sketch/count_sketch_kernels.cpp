// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/count_sketch_kernels.hpp"

#include <random>

#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace count_sketch {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              size_type sketch_size, array<IndexType>& hash_map,
              array<ValueType>& signs, uint64 seed)
{
    auto master = exec->get_master();
    auto input_size = hash_map.get_size();
    array<IndexType> host_hash{master, input_size};
    array<ValueType> host_signs{master, input_size};
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<IndexType> hash_dist(
        0, static_cast<IndexType>(sketch_size - 1));
    std::bernoulli_distribution sign_dist(0.5);
    for (size_type i = 0; i < input_size; ++i) {
        host_hash.get_data()[i] = hash_dist(rng);
        host_signs.get_data()[i] =
            sign_dist(rng) ? one<ValueType>() : -one<ValueType>();
    }
    hash_map = host_hash;
    signs = host_signs;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COUNT_SKETCH_GENERATE);


namespace kernel {


// Gather-based: each thread computes one output element x[out_row, col]
// by scanning all input rows that hash to out_row. No atomics needed,
// deterministic accumulation order matches the reference kernel.
template <typename DeviceValueType, typename IndexType>
__global__ void count_sketch_apply(const IndexType* __restrict__ hash_map,
                                   const DeviceValueType* __restrict__ signs,
                                   size_type input_size, size_type num_cols,
                                   const DeviceValueType* __restrict__ b,
                                   size_type b_stride,
                                   DeviceValueType* __restrict__ x,
                                   size_type x_stride, size_type x_rows)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto total_out = x_rows * num_cols;
    for (auto idx = tid; idx < total_out; idx += blockDim.x * gridDim.x) {
        auto out_row = idx / num_cols;
        auto col = idx % num_cols;
        auto acc = zero<DeviceValueType>();
        for (size_type i = 0; i < input_size; ++i) {
            if (hash_map[i] == static_cast<IndexType>(out_row)) {
                acc += signs[i] * b[i * b_stride + col];
            }
        }
        x[out_row * x_stride + col] = acc;
    }
}


// Gather-based rapply: each thread computes one output element x[row, out_col]
template <typename DeviceValueType, typename IndexType>
__global__ void count_sketch_rapply(const IndexType* __restrict__ hash_map,
                                    const DeviceValueType* __restrict__ signs,
                                    size_type input_size, size_type num_rows,
                                    const DeviceValueType* __restrict__ b,
                                    size_type b_stride,
                                    DeviceValueType* __restrict__ x,
                                    size_type x_stride, size_type x_cols)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto total_out = num_rows * x_cols;
    for (auto idx = tid; idx < total_out; idx += blockDim.x * gridDim.x) {
        auto row = idx / x_cols;
        auto out_col = idx % x_cols;
        auto acc = zero<DeviceValueType>();
        for (size_type i = 0; i < input_size; ++i) {
            if (hash_map[i] == static_cast<IndexType>(out_col)) {
                acc += signs[i] * b[row * b_stride + i];
            }
        }
        x[row * x_stride + out_col] = acc;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const array<IndexType>& hash_map, const array<ValueType>& signs,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<ValueType> x)
{
    constexpr int block_size = 256;
    auto input_size = hash_map.get_size();
    auto num_cols = b.size[1];
    auto total = input_size * num_cols;
    auto grid_size = static_cast<int>((total + block_size - 1) / block_size);
    if (grid_size > 0) {
        kernel::count_sketch_apply<<<grid_size, block_size, 0,
                                     exec->get_stream()>>>(
            hash_map.get_const_data(), as_device_type(signs.get_const_data()),
            input_size, num_cols, as_device_type(b.values), b.stride,
            as_device_type(x.values), x.stride, x.size[0]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COUNT_SKETCH_APPLY);


template <typename ValueType, typename IndexType>
void rapply(std::shared_ptr<const DefaultExecutor> exec,
            const array<IndexType>& hash_map, const array<ValueType>& signs,
            matrix::view::dense<const ValueType> b,
            matrix::view::dense<ValueType> x)
{
    constexpr int block_size = 256;
    auto input_size = hash_map.get_size();
    auto num_rows = b.size[0];
    auto total = num_rows * input_size;
    auto grid_size = static_cast<int>((total + block_size - 1) / block_size);
    if (grid_size > 0) {
        kernel::count_sketch_rapply<<<grid_size, block_size, 0,
                                      exec->get_stream()>>>(
            hash_map.get_const_data(), as_device_type(signs.get_const_data()),
            input_size, num_rows, as_device_type(b.values), b.stride,
            as_device_type(x.values), x.stride, x.size[1]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COUNT_SKETCH_RAPPLY);


}  // namespace count_sketch
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
