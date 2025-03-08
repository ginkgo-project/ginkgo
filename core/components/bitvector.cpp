// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector.hpp"

#include <ginkgo/core/base/math.hpp>

#include "core/components/bitvector_kernels.hpp"


namespace gko {
namespace {


GKO_REGISTER_OPERATION(compute_bits_and_ranks,
                       bitvector::compute_bits_and_ranks);


}  // namespace

template <typename IndexType>
device_bitvector<IndexType> bitvector<IndexType>::device_view() const
{
    return device_bitvector<IndexType>{bits_.get_const_data(),
                                       ranks_.get_const_data(), size_};
}


template <typename IndexType>
bitvector<IndexType>::bitvector(std::shared_ptr<const Executor> exec,
                                index_type size)
    : size_{size},
      bits_{exec, static_cast<size_type>(ceildiv(size, block_size))},
      ranks_{exec, static_cast<size_type>(ceildiv(size, block_size))}
{
    bits_.fill(storage_type{});
    ranks_.fill(0);
}


template <typename IndexType>
bitvector<IndexType> bitvector<IndexType>::from_sorted_indices(
    const array<IndexType>& indices, index_type size)
{
    const auto exec = indices.get_executor();
    bitvector result{exec, size};
    exec->run(make_compute_bits_and_ranks(
        indices.get_const_data(), static_cast<IndexType>(indices.get_size()),
        size, result.bits_.get_data(), result.ranks_.get_data()));
    return result;
}


#define GKO_DEFINE_BITVECTOR(IndexType) class bitvector<IndexType>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DEFINE_BITVECTOR);


}  // namespace gko
