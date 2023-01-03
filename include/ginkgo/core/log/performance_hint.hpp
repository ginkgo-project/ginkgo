/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_LOG_PERFORMANCE_HINT_HPP_
#define GKO_PUBLIC_CORE_LOG_PERFORMANCE_HINT_HPP_


#include <fstream>
#include <iostream>
#include <unordered_map>


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


/**
 * PerformanceHint is a Logger which analyzes the performance of the application
 * and outputs hints for unnecessary copies and allocations.
 * The specific patterns it checks for are:
 * - repeated cross-executor copies from or to the same pointer
 * - repeated allocation/free pairs of the same size
 *
 * @ingroup log
 */
class PerformanceHint : public Logger {
public:
    void on_allocation_completed(const Executor* exec,
                                 const size_type& num_bytes,
                                 const uintptr& location) const override;

    void on_free_completed(const Executor* exec,
                           const uintptr& location) const override;

    void on_copy_completed(const Executor* from, const Executor* to,
                           const uintptr& location_from,
                           const uintptr& location_to,
                           const size_type& num_bytes) const override;

    /**
     * Writes out the cross-executor writes and allocations that have been
     * stored so far.
     */
    void print_status() const;

    /**
     * Creates a PerformanceHint logger. This dynamically allocates the memory,
     * constructs the object and returns an std::unique_ptr to this object.
     *
     * @param os  the stream used for this logger
     * @param allocation_size_limit  ignore allocations below this limit (bytes)
     * @param copy_size_limit  ignore copies below this limit (bytes)
     * @param histogram_max_size  how many allocation sizes and/or pointers to
     *                            keep track of at most?
     *
     * @return an std::unique_ptr to the the constructed object
     */
    static std::unique_ptr<PerformanceHint> create(
        std::ostream& os = std::cerr, size_type allocation_size_limit = 16,
        size_type copy_size_limit = 16, size_type histogram_max_size = 1024)
    {
        return std::unique_ptr<PerformanceHint>(new PerformanceHint(
            os, allocation_size_limit, copy_size_limit, histogram_max_size));
    }

protected:
    explicit PerformanceHint(std::ostream& os, size_type allocation_size_limit,
                             size_type copy_size_limit,
                             size_type histogram_max_size)
        : Logger(mask_),
          os_(&os),
          allocation_size_limit_{allocation_size_limit},
          copy_size_limit_{copy_size_limit},
          histogram_max_size_{histogram_max_size}
    {}

private:
    // set a breakpoint here if you want to see where the output comes from!
    std::ostream& log() const;

    std::ostream* os_;
    mutable std::unordered_map<uintptr_t, size_type> allocation_sizes_;
    mutable std::unordered_map<size_type, int> allocation_histogram_;
    mutable std::unordered_map<uintptr_t, int> copy_src_histogram_;
    mutable std::unordered_map<uintptr_t, int> copy_dst_histogram_;
    size_type allocation_size_limit_;
    size_type copy_size_limit_;
    size_type histogram_max_size_;
    static constexpr Logger::mask_type mask_ =
        Logger::allocation_completed_mask | Logger::free_completed_mask |
        Logger::copy_completed_mask;
    static constexpr const char* prefix_ = "[PERFORMANCE] >>> ";
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_PERFORMANCE_HINT_HPP_
