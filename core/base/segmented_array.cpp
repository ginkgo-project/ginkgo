// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/segmented_array.hpp>


#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace {


GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum_nonnegative);


}


template <typename T>
size_type segmented_array<T>::get_size() const
{
    return buffer_.get_size();
}


template <typename T>
size_type segmented_array<T>::get_segment_count() const
{
    return offsets_.get_size() ? offsets_.get_size() - 1 : 0;
}


template <typename T>
T* segmented_array<T>::get_flat_data()
{
    return buffer_.get_data();
}


template <typename T>
const T* segmented_array<T>::get_const_flat_data() const
{
    return buffer_.get_const_data();
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


template <typename T>
segmented_array<T>::segmented_array(std::shared_ptr<const Executor> exec)
    : buffer_(exec), offsets_(exec, 1)
{
    offsets_.fill(0);
}


array<int64> sizes_to_offsets(const gko::array<int64>& sizes)
{
    auto exec = sizes.get_executor();
    array<int64> offsets(exec, sizes.get_size() + 1);
    exec->copy(sizes.get_size(), sizes.get_const_data(), offsets.get_data());
    exec->run(make_prefix_sum(offsets.get_data(), offsets.get_size()));
    return offsets;
}


template <typename T>
segmented_array<T> segmented_array<T>::create_from_sizes(
    const gko::array<int64>& sizes)
{
    return create_from_offsets(sizes_to_offsets(sizes));
}


template <typename T>
segmented_array<T> segmented_array<T>::create_from_sizes(
    gko::array<T> buffer, const gko::array<int64>& sizes)
{
    return create_from_offsets(std::move(buffer), sizes_to_offsets(sizes));
}


template <typename T>
segmented_array<T> segmented_array<T>::create_from_offsets(
    gko::array<int64> offsets)
{
    GKO_THROW_IF_INVALID(offsets.get_size() > 0,
                         "The offsets for segmented_arrays require at least "
                         "one element.");
    auto size =
        static_cast<size_type>(get_element(offsets, offsets.get_size() - 1));
    return create_from_offsets(array<T>{offsets.get_executor(), size},
                               std::move(offsets));
}


template <typename T>
segmented_array<T> segmented_array<T>::create_from_offsets(
    gko::array<T> buffer, gko::array<int64> offsets)
{
    GKO_ASSERT_EQ(buffer.get_size(),
                  get_element(offsets, offsets.get_size() - 1));
    segmented_array<T> result(buffer.get_executor());
    result.offsets_ = std::move(offsets);
    result.buffer_ = std::move(buffer);
    return result;
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
                                    const segmented_array& other)
    : segmented_array(exec)
{
    *this = other;
}


template <typename T>
segmented_array<T>::segmented_array(const segmented_array& other)
    : segmented_array(other.get_executor())
{
    *this = other;
}


template <typename T>
segmented_array<T>::segmented_array(segmented_array&& other)
    : segmented_array(other.get_executor())
{
    *this = std::move(other);
}


template <typename T>
segmented_array<T>& segmented_array<T>::operator=(const segmented_array& other)
{
    if (this != &other) {
        buffer_ = other.buffer_;
        offsets_ = other.offsets_;
    }
    return *this;
}


template <typename T>
segmented_array<T>& segmented_array<T>::operator=(segmented_array&& other)
{
    if (this != &other) {
        buffer_ = std::move(other.buffer_);
        offsets_ = std::exchange(other.offsets_,
                                 array<int64>{other.get_executor(), {0}});
    }
    return *this;
}


#define GKO_DECLARE_SEGMENTED_ARRAY(_type) class segmented_array<_type>

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SEGMENTED_ARRAY);


}  // namespace gko
