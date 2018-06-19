/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_STOP_STOPPING_STATUS_HPP_
#define GKO_CORE_STOP_STOPPING_STATUS_HPP_


#include "core/base/array.hpp"
#include "core/base/types.hpp"


namespace gko {


/**
 * This class is used to keep track of the stopping status of one vector.
 */
class stopping_status {
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
     * Call if a stop occured due to a hard limit (and convergence was not
     * reached).
     * @param id  id of the stopping criteria.
     * @param setFinalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    GKO_ATTRIBUTES GKO_INLINE void stop(uint8 id,
                                        bool setFinalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data_ |= (id & id_mask_);
            if (setFinalized) {
                data_ |= finalized_mask_;
            }
        }
    }

    /**
     * Call if convergence occured.
     * @param id  id of the stopping criteria.
     * @param setFinalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    GKO_ATTRIBUTES GKO_INLINE void converge(uint8 id,
                                            bool setFinalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data_ |= converged_mask_ | (id & id_mask_);
            if (setFinalized) {
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

    GKO_ATTRIBUTES GKO_INLINE void clear() noexcept { data_ = 0; }

private:
    static constexpr uint8 converged_mask_ = uint8{1} << 7;
    static constexpr uint8 finalized_mask_ = uint8{1} << 6;
    static constexpr uint8 id_mask_ = (uint8{1} << 6) - uint8{1};

    uint8 data_{};
};


}  // namespace gko


#endif  // GKO_CORE_STOP_STOPPING_STATUS_HPP_
