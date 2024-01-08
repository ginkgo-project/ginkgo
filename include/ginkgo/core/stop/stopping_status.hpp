// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_STOPPING_STATUS_HPP_
#define GKO_PUBLIC_CORE_STOP_STOPPING_STATUS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * This class is used to keep track of the stopping status of one vector.
 *
 * @ingroup stop
 */
class stopping_status {
    friend GKO_ATTRIBUTES GKO_INLINE bool operator==(
        const stopping_status& x, const stopping_status& y) noexcept;
    friend GKO_ATTRIBUTES GKO_INLINE bool operator!=(
        const stopping_status& x, const stopping_status& y) noexcept;

public:
    /**
     * Check if any stopping criteria was fulfilled.
     * @return Returns true if any stopping criteria was fulfilled.
     */
    GKO_ATTRIBUTES GKO_INLINE bool has_stopped() const noexcept
    {
        return get_id();
    }

    /**
     * Check if convergence was reached.
     * @return Returns true if convergence was reached.
     */
    GKO_ATTRIBUTES GKO_INLINE bool has_converged() const noexcept
    {
        return data_ & converged_mask_;
    }

    /**
     * Check if the corresponding vector stores the finalized result.
     * @return Returns true if the corresponding vector stores the finalized
     * result.
     */
    GKO_ATTRIBUTES GKO_INLINE bool is_finalized() const noexcept
    {
        return data_ & finalized_mask_;
    }

    /**
     * Get the id of the stopping criterion which caused the stop.
     * @return Returns the id of the stopping criterion which caused the stop.
     */
    GKO_ATTRIBUTES GKO_INLINE uint8 get_id() const noexcept
    {
        return data_ & id_mask_;
    }

    /**
     * Clear all flags.
     */
    GKO_ATTRIBUTES GKO_INLINE void reset() noexcept { data_ = uint8{0}; }

    /**
     * Call if a stop occurred due to a hard limit (and convergence was not
     * reached).
     * @param id  id of the stopping criteria.
     * @param set_finalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    GKO_ATTRIBUTES GKO_INLINE void stop(uint8 id,
                                        bool set_finalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data_ |= (id & id_mask_);
            if (set_finalized) {
                data_ |= finalized_mask_;
            }
        }
    }

    /**
     * Call if convergence occurred.
     * @param id  id of the stopping criteria.
     * @param set_finalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    GKO_ATTRIBUTES GKO_INLINE void converge(uint8 id,
                                            bool set_finalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data_ |= converged_mask_ | (id & id_mask_);
            if (set_finalized) {
                data_ |= finalized_mask_;
            }
        }
    }

    /**
     * Set the result to be finalized (it needs to be stopped or converged
     * first).
     */
    GKO_ATTRIBUTES GKO_INLINE void finalize() noexcept
    {
        if (this->has_stopped()) {
            data_ |= finalized_mask_;
        }
    }

private:
    static constexpr uint8 converged_mask_ = uint8{1} << 7;
    static constexpr uint8 finalized_mask_ = uint8{1} << 6;
    static constexpr uint8 id_mask_ = (uint8{1} << 6) - uint8{1};

    uint8 data_;
};


/**
 * Checks if two stopping statuses are equivalent.
 *
 * @param x a stopping status
 * @param y a stopping status
 *
 * @return true if and only if both `x` and `y` have the same mask and converged
 *         and finalized state
 */
GKO_ATTRIBUTES GKO_INLINE bool operator==(const stopping_status& x,
                                          const stopping_status& y) noexcept
{
    return x.data_ == y.data_;
}


/**
 * Checks if two stopping statuses are different.
 *
 * @param x a stopping status
 * @param y a stopping status
 *
 * @return true if and only if `!(x == y)`
 */
GKO_ATTRIBUTES GKO_INLINE bool operator!=(const stopping_status& x,
                                          const stopping_status& y) noexcept
{
    return x.data_ != y.data_;
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_STOPPING_STATUS_HPP_
