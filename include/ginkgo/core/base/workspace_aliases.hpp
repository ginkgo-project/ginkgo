// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_WORKSPACE_ALIASES_HPP_
#define GKO_PUBLIC_CORE_BASE_WORKSPACE_ALIASES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>


// This code is a modified version of the code from CCCL
// (https://github.com/NVIDIA/cccl) (cub/detail/temporary_storage.cuh and
// cub/temporary_storage.cuh), made available through the Apache-2.0 and BSD-3
// licenses. See ABOUT-LICENSING.md for more details.


namespace gko {
namespace detail {


template <int num_allocs>
GKO_ATTRIBUTES GKO_INLINE GKO_DEVICE_ERROR_TYPE create_workspace_aliases(
    void* workspace_ptr, size_t& num_bytes, void* (&allocations)[num_allocs],
    size_t (&allocation_sizes)[num_allocs])
{
    constexpr int align_bytes = 8;
    constexpr int align_mask = ~(align_bytes - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[num_allocs];
    size_t bytes_needed = 0;
    for (int i = 0; i < num_allocs; ++i) {
        size_t allocation_bytes =
            (allocation_sizes[i] + align_bytes - 1) & align_mask;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += align_bytes - 1;

    // Check if the caller is simply requesting the size of the storage
    // allocation
    if (!workspace_ptr) {
        num_bytes = bytes_needed;
        return GKO_DEVICE_NO_ERROR;
    }

    // Check if enough storage provided
    if (num_bytes < bytes_needed) {
        return GKO_DEVICE_ERROR_INVALID;
    }

    // Alias
    workspace_ptr =
        (void*)((size_t(workspace_ptr) + align_bytes - 1) & align_mask);
    for (int i = 0; i < num_allocs; ++i) {
        allocations[i] =
            static_cast<char*>(workspace_ptr) + allocation_offsets[i];
    }

    return GKO_DEVICE_NO_ERROR;
}


class slot;

template <typename T>
class alias;

template <int num_slots>
class layout;

class slot {
    template <typename T>
    friend class alias;

    template <int>
    friend class layout;

public:
    slot() = default;

    /**
     * @brief Returns an array of type @p T and length @p num_elems
     */
    template <typename T>
    GKO_ATTRIBUTES alias<T> create_alias(std::size_t num_elems = 0);

private:
    GKO_ATTRIBUTES void set_bytes_required(std::size_t new_size)
    {
        size_ = max(size_, new_size);
    }

    GKO_ATTRIBUTES std::size_t get_bytes_required() const { return size_; }

    GKO_ATTRIBUTES void set_storage(void* ptr) { ptr_ = ptr; }

    GKO_ATTRIBUTES void* get_storage() const { return ptr_; }

    std::size_t size_{};

    void* ptr_{};
};

/**
 * @brief Named memory region of a temporary storage slot
 *
 * @par Overview
 * This class provides a typed wrapper of a temporary slot memory region.
 * It can be considered as a field in the C++ union. It's only possible to
 * increase the array size.
 */
template <typename T>
class alias {
    friend class slot;

public:
    alias() = delete;

    /**
     * @brief Returns pointer to array
     *
     * If the @p num_elems number is equal to zero, or storage layout isn't
     * mapped,
     * @p nullptr is returned.
     */
    GKO_ATTRIBUTES T* get() const
    {
        if (num_elems_ == 0) {
            return nullptr;
        }

        return reinterpret_cast<T*>(slot_.get_storage());
    }

private:
    GKO_ATTRIBUTES explicit alias(slot& slot, std::size_t num_elems = 0)
        : slot_(slot), num_elems_(num_elems)
    {
        this->update_slot();
    }

    GKO_ATTRIBUTES void update_slot()
    {
        slot_.set_bytes_required(num_elems_ * sizeof(T));
    }
    slot& slot_;
    std::size_t num_elems_{};
};


template <typename T>
GKO_ATTRIBUTES alias<T> slot::create_alias(std::size_t num_elems)
{
    return alias<T>(*this, num_elems);
}


/**
 * @brief Temporary storage layout represents a structure with
 *        @p num_slots union-like fields
 *
 * The layout can be mapped to a temporary buffer only once.
 *
 * @par A Simple Example
 * @code
 * gko::detail::layout<2> temp;
 *
 * auto slot_1 = temp.get_slot(0);
 * auto slot_2 = temp.get_slot(1);
 *
 * // Add fields into the first slot
 * auto int_array = slot_1->create_alias<int>(1);
 * auto double_array = slot_2->create_alias<double>(2);
 *
 * temporary_storage.map_to_buffer(workspace_ptr, num_bytes);
 *
 * // Use pointers
 * int *int_ptr = int_array.get();
 * double *double_ptr = double_array.get();
 * @endcode
 */
template <int num_slots>
class layout {
public:
    layout() = default;

    GKO_ATTRIBUTES slot* get_slot(int slot_id)
    {
        if (slot_id < num_slots) {
            return &slots_[slot_id];
        }

        return nullptr;
    }

    /**
     * @brief Maps the layout to the temporary storage buffer.
     */
    GKO_ATTRIBUTES GKO_DEVICE_ERROR_TYPE map_to_buffer(void* workspace_ptr,
                                                       std::size_t num_bytes)
    {
        if (is_layout_mapped_) {
            return GKO_DEVICE_ERROR_INVALID;  // TODO: maybe use something
                                              // similar to
                                              // cudaErrorAlreadyMapped
        }

        this->initialize();

        GKO_DEVICE_ERROR_TYPE error = GKO_DEVICE_NO_ERROR;
        if ((error = create_workspace_aliases(workspace_ptr, num_bytes,
                                              data_ptrs_, slot_sizes_))) {
            return error;
        }

        for (std::size_t slot_id = 0; slot_id < num_slots; slot_id++) {
            slots_[slot_id].set_storage(data_ptrs_[slot_id]);
        }

        is_layout_mapped_ = true;
        return error;
    }

private:
    GKO_ATTRIBUTES void initialize()
    {
        if (is_layout_mapped_) {
            return;
        }

        for (std::size_t slot_id = 0; slot_id < num_slots; slot_id++) {
            const std::size_t slot_size = slots_[slot_id].get_bytes_required();

            slot_sizes_[slot_id] = slot_size;
            data_ptrs_[slot_id] = nullptr;
        }
    }
    slot slots_[num_slots];
    std::size_t slot_sizes_[num_slots];
    void* data_ptrs_[num_slots];
    bool is_layout_mapped_{};
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_WORKSPACE_ALIASES_HPP_
