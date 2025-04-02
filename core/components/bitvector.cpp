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
    return device_bitvector<IndexType>{this->get_bits(), this->get_ranks(),
                                       this->get_size()};
}


template <typename IndexType>
std::shared_ptr<const Executor> bitvector<IndexType>::get_executor() const
{
    return bits_.get_executor();
}


template <typename IndexType>
const typename bitvector<IndexType>::storage_type*
bitvector<IndexType>::get_bits() const
{
    return bits_.get_const_data();
}


template <typename IndexType>
const IndexType* bitvector<IndexType>::get_ranks() const
{
    return ranks_.get_const_data();
}


template <typename IndexType>
IndexType bitvector<IndexType>::get_size() const
{
    return size_;
}


template <typename IndexType>
IndexType bitvector<IndexType>::get_num_blocks() const
{
    return static_cast<IndexType>(ceildiv(this->get_size(), block_size));
}


template <typename IndexType>
bitvector<IndexType>::bitvector(array<storage_type> bits,
                                array<index_type> ranks, index_type size)
    : size_{size}, bits_{std::move(bits)}, ranks_{std::move(ranks)}
{
    GKO_ASSERT(bits_.get_executor() == ranks_.get_executor());
    GKO_ASSERT(this->get_num_blocks() == bits_.get_size());
    GKO_ASSERT(this->get_num_blocks() == ranks_.get_size());
}


template <typename IndexType>
bitvector<IndexType>::bitvector(std::shared_ptr<const Executor> exec,
                                index_type size)
    : size_{size},
      bits_{exec, static_cast<size_type>(this->get_num_blocks())},
      ranks_{exec, static_cast<size_type>(this->get_num_blocks())}
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
