// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>


namespace gko {

template <typename T>
struct segmented_array {
    using value_type = T;
    using reference = gko::array<T>&;
    using const_reference = const gko::array<T>&;

    segmented_array() = default;

    explicit segmented_array(std::shared_ptr<const Executor> exec);

    segmented_array(std::shared_ptr<const Executor> exec,
                    const segmented_array& other);

    segmented_array(std::shared_ptr<const Executor> exec,
                    segmented_array&& other);

    segmented_array(std::shared_ptr<const Executor> exec,
                    const std::vector<int64>& sizes);

    segmented_array(gko::array<T> buffer, const std::vector<int64>& sizes);

    size_type size() const;

    reference get_flat();

    const_reference get_flat() const;

    const gko::array<int64>& get_offsets() const;

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
            return std::make_unique<segmented_array<T>>(exec, *ptr);
        } else {
            return std::make_unique<segmented_array<T>>(std::move(exec),
                                                        std::move(*ptr));
        }
    }
};

template <typename T>
struct temporary_clone_helper<const segmented_array<T>> {
    static std::unique_ptr<const segmented_array<T>> create(
        std::shared_ptr<const Executor> exec, const segmented_array<T>* ptr,
        bool)
    {
        return std::make_unique<segmented_array<T>>(exec, *ptr);
    }
};


// specialization for non-constant arrays, copying back via assignment
template <typename T>
class copy_back_deleter<segmented_array<T>> {
public:
    using pointer = segmented_array<T>*;

    /**
     * Creates a new deleter object.
     *
     * @param original  the origin object where the data will be copied before
     *                  deletion
     */
    copy_back_deleter(pointer original) : original_{original} {}

    /**
     * Copies back the pointed-to object to the original and deletes it.
     *
     * @param ptr  pointer to the object to be copied back and deleted
     */
    void operator()(pointer ptr) const
    {
        *original_ = *ptr;
        delete ptr;
    }

private:
    pointer original_;
};


}  // namespace detail
}  // namespace gko
