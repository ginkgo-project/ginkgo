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

#ifndef GKO_PUBLIC_CORE_BASE_OVERLAP_HPP_
#define GKO_PUBLIC_CORE_BASE_OVERLAP_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


template <typename ValueType>
struct Overlap {
    Array<ValueType> get_overlaps() const { return overlaps_; }
    Array<bool> get_unidirectional_array() const { return is_unidirectional_; }
    size_type get_num_elems() const { return overlaps_.get_num_elems(); }

    const ValueType &get_overlap() const
    {
        GKO_ASSERT(overlaps_.get_num_elems() == 1);
        return overlaps_.get_const_data()[0];
    }

    const bool &is_unidirectional() const
    {
        GKO_ASSERT(is_unidirectional_.get_num_elems() == 1);
        return is_unidirectional_.get_const_data()[0];
    }

    void set_executor(std::shared_ptr<const Executor> exec)
    {
        is_unidirectional_.set_executor(exec);
        overlaps_.set_executor(exec);
    }

    Overlap() noexcept : is_unidirectional_{}, overlaps_{} {}

    Overlap(std::shared_ptr<const Executor> exec)
        : is_unidirectional_{}, overlaps_{}
    {}

    Overlap(std::shared_ptr<const Executor> exec, ValueType overlap,
            bool is_unidirectional)
        : is_unidirectional_{exec, {is_unidirectional}},
          overlaps_{exec, {overlap}}
    {}

    template <typename OverlapArray, typename UnidirArray>
    Overlap(std::shared_ptr<const Executor> exec, OverlapArray &&overlap,
            UnidirArray &&is_unidirectional)
        : is_unidirectional_{exec, std::forward<OverlapArray>(overlap)},
          overlaps_{exec, std::forward<UnidirArray>(is_unidirectional)}
    {}

private:
    Array<bool> is_unidirectional_;
    Array<ValueType> overlaps_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_OVERLAP_HPP_
