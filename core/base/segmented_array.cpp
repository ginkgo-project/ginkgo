// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/segmented_array.hpp>


#include <numeric>

namespace gko {


template <typename T>
segmented_array<T>::segmented_array(std::shared_ptr<const Executor> exec)
    : buffer_(exec), offsets_(exec, 1)
{
    offsets_.fill(0);
}


template <typename T>
segmented_array<T>::segmented_array(std::shared_ptr<const Executor> exec,
                                    const segmented_array& other)
    : segmented_array(exec)
{
    *this = other;
}


template <typename T>
segmented_array<T>::segmented_array(std::shared_ptr<const Executor> exec,
                                    segmented_array&& other)
    : segmented_array(exec)
{
    *this = std::move(other);
}


template <typename T>
segmented_array<T>::segmented_array(std::shared_ptr<const Executor> exec,
                                    const std::vector<int64>& sizes)
    : segmented_array(
          gko::array<T>(exec,
                        std::accumulate(sizes.begin(), sizes.end(), int64{})),
          sizes)
{}


template <typename T>
segmented_array<T>::segmented_array(array<T> buffer,
                                    const std::vector<int64>& sizes)
    : buffer_(std::move(buffer))
{
    auto exec = buffer_.get_executor();
    offsets_ = gko::array<int64>(exec->get_master(), sizes.size() + 1);
    offsets_.fill(0);
    std::partial_sum(sizes.begin(), sizes.end(), offsets_.get_data() + 1);
    buffer_.resize_and_reset(offsets_.get_data()[sizes.size()]);
    offsets_.set_executor(exec);
}


template <typename T>
size_type segmented_array<T>::size() const
{
    return offsets_.get_size() ? offsets_.get_size() - 1 : 0;
}


template <typename T>
array<T>& segmented_array<T>::get_flat()
{
    return buffer_;
}


template <typename T>
const array<T>& segmented_array<T>::get_flat() const
{
    return buffer_;
}


template <typename T>
const gko::array<int64>& segmented_array<T>::get_offsets() const
{
    return offsets_;
}


template <typename T>
std::shared_ptr<const Executor> segmented_array<T>::get_executor() const
{
    return buffer_.get_executor();
}


#define GKO_DECLARE_SEGMENTED_ARRAY(_vtype) class segmented_array<_vtype>

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SEGMENTED_ARRAY);

#undef GKO_DECLARE_SEGMENTED_ARRAY


}  // namespace gko
