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


#ifndef GKO_CORE_SOLVER_STOPPING_STATUS_HPP_
#define GKO_CORE_SOLVER_STOPPING_STATUS_HPP_

#include "core/base/array.hpp"
#include "core/base/types.hpp"

namespace gko {

/**
 * This structure is used to keep track of the stopping status of one vector.
 */
struct stopping_status {
public:
    stopping_status() = default;

    /**
     * Check if any stopping criteria was fulfilled.
     * @return Returns true if any stopping criteria was fulfilled.
     */
    bool has_stopped() const noexcept { return get_id(); }

    /**
     * Check if convergence was reached.
     * @return Returns true if convergence was reached.
     */
    bool has_converged() const noexcept { return data & converged_mask; }

    /**
     * Check if the corresponding vector stores the finalized result.
     * @return Returns true if the corresponding vector stores the finalized
     * result.
     */
    bool is_finalized() const noexcept { return data & finalized_mask; }

    /**
     * Get the id of the stopping criterion which caused the stop.
     * @return Returns the id of the stopping criterion which caused the stop.
     */
    uint8 get_id() const noexcept { return data & id_mask; }

    /**
     * Clear all flags.
     */
    void reset() noexcept { data = uint8{0}; }

    /**
     * Call if a stop occured due to a hard limit (and convergence was not
     * reached).
     * @param id  id of the stopping criteria.
     * @param setFinalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    void stop(uint8 id, bool setFinalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data |= (id & id_mask);
            if (setFinalized) {
                data |= finalized_mask;
            }
        }
    }

    /**
     * Call if convergence occured.
     * @param id  id of the stopping criteria.
     * @param setFinalized  Controls if the current version should count as
     * finalized (set to true) or not (set to false).
     */
    void converge(uint8 id, bool setFinalized = true) noexcept
    {
        if (!this->has_stopped()) {
            data |= converged_mask | (id & id_mask);
            if (setFinalized) {
                data |= finalized_mask;
            }
        }
    }

    /**
     * Set the result to be finalized (it needs to be stopped or converged
     * first).
     */
    void finalize() noexcept
    {
        if (this->has_stopped()) {
            data |= finalized_mask;
        }
    }


    static constexpr uint8 converged_mask = uint8{1} << 7;
    static constexpr uint8 finalized_mask = uint8{1} << 6;
    static constexpr uint8 id_mask = (uint8{1} << 6) - uint8{1};

private:
    uint8 data;
};

}  // namespace gko

#endif  // GKO_CORE_SOLVER_STOPPING_STATUS_HPP_
