// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_FRSZ2_HPP_
#define GKO_ACCESSOR_FRSZ2_HPP_


#include <frsz2.hpp>


#include <array>
#include <cinttypes>
#include <memory>
#include <type_traits>
#include <utility>


#include "accessor_helper.hpp"
#include "index_span.hpp"
#include "range.hpp"
#include "utils.hpp"


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace acc {


namespace pattern {


// Might need a work-around to not have a size of 1 Byte for 1D
template <int dimensionality_, typename IndexType = size_type>
struct row_major {
    static constexpr int dimensionality = dimensionality_;
    using index_type = IndexType;

    static_assert(dimensionality >= 1,
                  "This class only supports dimensionality >= 1.");
    template <typename... Indices>
    index_type get_linear_index(Indices&&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::detail::row_major_helper_s<
            index_type, dimensionality>::compute(stride_,
                                                 std::forward<Indices>(
                                                     indices)...);
    }

private:
    std::array<index_type, dimensionality - 1> stride_;
};


// Careful, this is neither row- nor column-major!
template <typename IndexType = std::int32_t>
struct cb_gmres {
public:
    static constexpr int dimensionality = 3;
    using index_type = IndexType;

    GKO_ACC_ATTRIBUTES cb_gmres(std::array<index_type, 2> stride)
        : stride_{stride[0], stride[1]}
    {}

    GKO_ACC_ATTRIBUTES index_type get_linear_index(index_type krylov_vec,
                                                   index_type vec_idx,
                                                   index_type vec_rhs) const
    {
        return vec_rhs * stride_[0] + krylov_vec * stride_[1] + vec_idx;
    }

    GKO_ACC_ATTRIBUTES const index_type* get_stride() const { return stride_; }

private:
    index_type stride_[2];
};


}  // namespace pattern


template <int bits_per_value_, int max_exp_block_size_, typename ArithmeticType>
class frsz2 {
public:
    using access_pattern = pattern::cb_gmres<std::int32_t>;
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    static constexpr auto dimensionality = access_pattern::dimensionality;
    static constexpr auto bits_per_value{bits_per_value_};
    static constexpr auto max_exp_block_size{max_exp_block_size_};

    using dim_type = std::array<size_type, dimensionality>;

    static_assert(dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    // Normal operator() is not yet supported, so no `range` support

private:
    using index_type = access_pattern::index_type;
    using frsz2_compressor =
        frsz::frsz2_compressor<bits_per_value, max_exp_block_size,
                               arithmetic_type>;

    dim_type size_;
    access_pattern acc_pattern_;
    frsz2_compressor compressor_;

public:
    static GKO_ACC_ATTRIBUTES std::array<index_type, 2> get_required_stride(
        dim_type size)
    {
        const auto stride_1 = (size[1] / max_exp_block_size +
                               int(size[1] % max_exp_block_size > 0)) *
                              max_exp_block_size;
        return {size[0] * stride_1, stride_1};
    }

    // Returns the total number of elements that will be required for the FRSZ2
    // compression with the given size
    static GKO_ACC_ATTRIBUTES std::size_t num_elements_required(dim_type size)
    {
        return size[2] * get_required_stride(size)[0];
    }

    // Returns the size in Byte that `size` requires for the FRSZ2 compression
    static GKO_ACC_ATTRIBUTES std::size_t memory_requirement(dim_type size)
    {
        return frsz2_compressor::compute_compressed_memory_size_byte(
            num_elements_required(size));
    }

    /**
     * Creates an FRSZ2 accessor for an already allocated storage space.
     *
     * @param size  multidimensional size of the memory
     * @param storage  pointer to the block of memory containing the storage
     */
    constexpr GKO_ACC_ATTRIBUTES frsz2(dim_type size, std::uint8_t* storage)
        : size_(size),
          acc_pattern_(get_required_stride(size)),
          compressor_(storage, num_elements_required(size))
    {}

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    constexpr GKO_ACC_ATTRIBUTES frsz2() : frsz2{{0, 0, 0}, nullptr} {}

public:
    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns  length in dimension `dimension`
     */
    constexpr GKO_ACC_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }

    /**
     * Returns the stored value for the given indices.
     */
    constexpr GKO_ACC_ATTRIBUTES arithmetic_type read_element(
        index_type krylov_vec, index_type vec_idx, index_type vec_rhs) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // if (krylov_vec >= size_[0] || vec_idx >= get_stride()[1] ||
        //     vec_rhs >= size_[2]) {
        //     printf(
        //         "b %d (%d), t %d (%d): illegal read access: kv %lld / %lld,
        //         vi "
        //         "%lld / %lld (%lld), vr %lld / %lld\n",
        //         int(blockIdx.x), int(gridDim.x), int(threadIdx.x),
        //         int(blockDim.x), std::int64_t(krylov_vec),
        //         std::int64_t(size_[0]), std::int64_t(vec_idx),
        //         std::int64_t(size_[1]), std::int64_t(get_stride()[1]),
        //         std::int64_t(vec_rhs), std::int64_t(size_[2]));
        // }
        return compressor_.decompress_gpu_element(
            acc_pattern_.get_linear_index(krylov_vec, vec_idx, vec_rhs));
#else
        return compressor_.decompress_cpu_element(
            acc_pattern_.get_linear_index(krylov_vec, vec_idx, vec_rhs));
#endif
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    template <int block_size>
    __device__ void compress_gpu_function(index_type krylov_vec,
                                          index_type vec_idx,
                                          index_type vec_rhs,
                                          const arithmetic_type fp_input_value)
    {
        // if (krylov_vec >= size_[0] || vec_idx >= get_stride()[1] ||
        //     vec_rhs >= size_[2]) {
        //     printf(
        //         "b %d (%d), t %d (%d): illegal read access: kv %lld / %lld,
        //         vi "
        //         "%lld / %lld (%lld), vr %lld / %lld\n",
        //         int(blockIdx.x), int(gridDim.x), int(threadIdx.x),
        //         int(blockDim.x), std::int64_t(krylov_vec),
        //         std::int64_t(size_[0]), std::int64_t(vec_idx),
        //         std::int64_t(size_[1]), std::int64_t(get_stride()[1]),
        //         std::int64_t(vec_rhs), std::int64_t(size_[2]));
        // }
        compressor_.template compress_gpu_function<block_size>(
            acc_pattern_.get_linear_index(krylov_vec, vec_idx, vec_rhs),
            fp_input_value);
    }
#endif


    // TODO add the write for CPU

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    constexpr GKO_ACC_ATTRIBUTES dim_type get_size() const { return size_; }
    constexpr GKO_ACC_ATTRIBUTES auto get_stride() const
    {
        return acc_pattern_.get_stride();
    }
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_FRSZ2_HPP_
