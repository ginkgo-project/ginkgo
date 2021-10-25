/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_


#include <iostream>
#include <vector>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * A type representing the dimensions of a multidimensional batch object.
 *
 * @tparam Dimensionality  number of dimensions of the object
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @ingroup batch_dim
 */
template <size_type Dimensionality = 2, typename DimensionType = size_type>
struct batch_dim {
    static constexpr size_type dimensionality = Dimensionality;
    using dimension_type = DimensionType;

    /**
     * Checks if the batch_dim object stores equal sizes.
     *
     * @return bool representing whether equal sizes are being stored
     */
    bool stores_equal_sizes() const { return equal_sizes_; }

    /**
     * Get the number of batch entries stored
     *
     * @return num_batch_entries
     */
    size_type get_num_batch_entries() const { return num_batch_entries_; }

    /**
     * Get the sizes of all entries as a std::vector.
     *
     * @return  the std::vector of batch sizes
     */
    std::vector<dim<dimensionality, dimension_type>> get_batch_sizes() const
    {
        if (equal_sizes_) {
            if (num_batch_entries_ > 0) {
                return std::vector<dim<dimensionality, dimension_type>>(
                    num_batch_entries_, common_size_);
            } else {
                return std::vector<dim<dimensionality, dimension_type>>{
                    common_size_};
            }
        } else {
            return sizes_;
        }
    }

    /**
     * Get the batch size at a particular index.
     *
     * @param batch_entry  the index of the entry whose size is needed
     *
     * @return  the size of the batch entry at the requested batch-index
     */
    const dim<dimensionality, dimension_type>& at(
        const size_type batch_entry = 0) const
    {
        if (equal_sizes_) {
            return common_size_;
        } else {
            GKO_ASSERT(batch_entry < num_batch_entries_);
            return sizes_[batch_entry];
        }
    }

    /**
     * Checks if two batch_dim objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend bool operator==(const batch_dim& x, const batch_dim& y)
    {
        if (x.equal_sizes_ && y.equal_sizes_) {
            return x.num_batch_entries_ == y.num_batch_entries_ &&
                   x.common_size_ == y.common_size_;
        } else {
            return x.sizes_ == y.sizes_;
        }
    }

    /**
     * Creates a batch_dim object which stores a uniform size for all batch
     * entries.
     *
     * @param num_batch_entries  number of batch entries to be stored
     * @param common_size  the common size of all the batch entries stored
     *
     * @note  Use this constructor when uniform batches need to be stored.
     */
    batch_dim(const size_type num_batch_entries = 0,
              const dim<dimensionality, dimension_type>& common_size =
                  dim<dimensionality, dimension_type>{})
        : equal_sizes_(true),
          common_size_(common_size),
          num_batch_entries_(num_batch_entries),
          sizes_()
    {}

    /**
     * Creates a batch_dim object which stores possibly non-uniform sizes for
     * the different batch entries.
     *
     * @param batch_sizes  the std::vector object that stores the batch_sizes
     *
     * @note  Use this constructor when non-uniform batches need to be stored.
     */
    batch_dim(
        const std::vector<dim<dimensionality, dimension_type>>& batch_sizes)
        : equal_sizes_(false),
          common_size_(dim<dimensionality, dimension_type>{}),
          num_batch_entries_(batch_sizes.size()),
          sizes_(batch_sizes)
    {
        check_size_equality();
    }

private:
    void check_size_equality()
    {
        for (size_type i = 0; i < num_batch_entries_; ++i) {
            if (!(sizes_[i] == sizes_[0])) {
                return;
            }
        }
        common_size_ = sizes_[0];
        equal_sizes_ = true;
    }

    bool equal_sizes_{};
    size_type num_batch_entries_{};
    dim<dimensionality, dimension_type> common_size_{};
    std::vector<dim<dimensionality, dimension_type>> sizes_{};
};


/**
 * Checks if two batch dim objects are different.
 *
 * @tparam Dimensionality  number of dimensions of the dim objects
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param x  first object
 * @param y  second object
 *
 * @return `!(x == y)`
 */
template <size_type Dimensionality, typename DimensionType>
inline bool operator!=(const batch_dim<Dimensionality, DimensionType>& x,
                       const batch_dim<Dimensionality, DimensionType>& y)
{
    return !(x == y);
}


/**
 * Returns a batch_dim object with its dimensions swapped for batched operators
 *
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param dimensions original object
 *
 * @return a batch_dim object with the individual batches having their
 *         dimensions swapped
 */
template <typename DimensionType>
inline batch_dim<2, DimensionType> transpose(
    const batch_dim<2, DimensionType>& input)
{
    batch_dim<2, DimensionType> out{};
    if (input.stores_equal_sizes()) {
        out = batch_dim<2, DimensionType>(input.get_num_batch_entries(),
                                          gko::transpose(input.at(0)));
        return out;
    }
    auto trans =
        std::vector<dim<2, DimensionType>>(input.get_num_batch_entries());
    for (size_type i = 0; i < trans.size(); ++i) {
        trans[i] = transpose(input.at(i));
    }
    return batch_dim<2, DimensionType>(trans);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_DIM_HPP_
