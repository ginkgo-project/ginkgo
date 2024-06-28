// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <numeric>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>


namespace gko {

/**
 * \brief A minimal interface for a segmented array.
 *
 * The segmented array is stored as a flat buffer with an offsets array.
 * The segment `i` contains the index range `[offset[i], offset[i + 1])` of the
 * flat buffer.
 *
 * \tparam T value type stored in the arrays
 */
template <typename T>
struct segmented_array {
    /**
     * Create an empty segmented array
     *
     * @param exec  executor for storage arrays
     */
    explicit segmented_array(std::shared_ptr<const Executor> exec);

    /**
     * Creates an uninitialized segmented array with predefined segment sizes.
     *
     * @param exec  executor for storage arrays
     * @param sizes  the sizes of each segment
     */
    static segmented_array create_from_sizes(const gko::array<int64>& sizes);

    /**
     * Creates a segmented array from a flat buffer and segment sizes.
     *
     * @param buffer  the flat buffer whose size has to match the sum of sizes
     * @param sizes  the sizes of each segment
     */
    static segmented_array create_from_sizes(gko::array<T> buffer,
                                             const gko::array<int64>& sizes);

    /**
     * Creates an uninitialized segmented array from offsets.
     *
     * @param offsets  the index offsets for each segment, and the total size of
     *                 the buffer as last element
     */
    static segmented_array create_from_offsets(gko::array<int64> offsets);

    /**
     * Creates a segmented array from a flat buffer and offsets.
     *
     * @param buffer  the flat buffer whose size has to match the last element
     *                of offsets
     * @param offsets  the index offsets for each segment, and the total size of
     *                 the buffer as last element
     */
    static segmented_array create_from_offsets(gko::array<T> buffer,
                                               gko::array<int64> offsets);

    /**
     * Copies a segmented array to a different executor.
     *
     * @param exec  the executor to copy to
     * @param other  the segmented array to copy from
     */
    segmented_array(std::shared_ptr<const Executor> exec,
                    const segmented_array& other);

    /**
     * Moves a segmented array to a different executor.
     *
     * @param exec  the executor to move to
     * @param other  the segmented array to move from
     */
    segmented_array(std::shared_ptr<const Executor> exec,
                    segmented_array&& other);

    segmented_array(const segmented_array& other);

    segmented_array(segmented_array&& other) noexcept(false);

    segmented_array& operator=(const segmented_array& other);

    segmented_array& operator=(segmented_array&&) noexcept(false);

    /**
     * Get the total size of the stored buffer.
     *
     * @return  the total size of the stored buffer.
     */
    size_type get_size() const;

    /**
     * Get the number of segments.
     *
     * @return  the number of segments
     */
    size_type get_segment_count() const;

    /**
     * Access to the flat buffer.
     *
     * @return  the flat buffer
     */
    T* get_flat_data();

    /**
     * Const-access to the flat buffer
     *
     * @return  the flat buffer
     */
    const T* get_const_flat_data() const;

    /**
     * Access to the segment offsets.
     *
     * @return  the segment offsets
     */
    const gko::array<int64>& get_offsets() const;

    /**
     * Access the executor.
     *
     * @return  the executor
     */
    std::shared_ptr<const Executor> get_executor() const;

private:
    gko::array<T> buffer_;
    gko::array<int64> offsets_;
};


namespace detail {


template <typename T>
struct temporary_clone_helper<segmented_array<T>> {
    static std::unique_ptr<segmented_array<T>> create(
        std::shared_ptr<const Executor> exec, segmented_array<T>* ptr,
        bool copy_data)
    {
        if (copy_data) {
            return std::make_unique<segmented_array<T>>(
                make_array_view(exec, ptr->get_size(), ptr->get_flat_data()),
                ptr->get_offsets());
        } else {
            return std::make_unique<segmented_array<T>>(std::move(exec),
                                                        ptr->get_offsets());
        }
    }
};

template <typename T>
struct temporary_clone_helper<const segmented_array<T>> {
    static std::unique_ptr<const segmented_array<T>> create(
        std::shared_ptr<const Executor> exec, const segmented_array<T>* ptr,
        bool)
    {
        return std::make_unique<segmented_array<T>>(
            make_array_view(exec, ptr->get_size(), ptr->get_const_flat_data()),
            ptr->get_offsets());
    }
};


template <typename T>
class copy_back_deleter<segmented_array<T>>
    : public copy_back_deleter_from_assignment<segmented_array<T>> {
public:
    using copy_back_deleter_from_assignment<
        segmented_array<T>>::copy_back_deleter_from_assignment;
};


}  // namespace detail
}  // namespace gko
