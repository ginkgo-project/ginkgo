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
class Overlap {
public:
    const ValueType *get_overlaps() const { return overlaps_.get_const_data(); }

    const bool *get_unidirectional_array() const
    {
        return is_unidirectional_.get_const_data();
    }

    const bool *get_overlap_at_start_array() const
    {
        return overlap_at_start_.get_const_data();
    }

    size_type get_num_elems() const { return overlaps_.get_num_elems(); }

    const ValueType &get_overlap() const
    {
        return overlaps_.get_const_data()[0];
    }

    const bool &is_unidirectional() const
    {
        return is_unidirectional_.get_const_data()[0];
    }

    const bool &is_overlap_at_start() const
    {
        return overlap_at_start_.get_const_data()[0];
    }

    std::shared_ptr<const Executor> get_executor() const
    {
        return overlaps_.get_executor();
    }

    void set_executor(std::shared_ptr<const Executor> exec)
    {
        is_unidirectional_.set_executor(exec);
        overlaps_.set_executor(exec);
        overlap_at_start_.set_executor(exec);
    }

    Overlap() noexcept : is_unidirectional_{}, overlaps_{}, overlap_at_start_{}
    {}

    Overlap(std::shared_ptr<const Executor> exec)
        : is_unidirectional_{}, overlaps_{}, overlap_at_start_{}
    {}

    explicit Overlap(std::shared_ptr<const Executor> exec, size_type num_blocks,
                     ValueType overlap, bool is_unidirectional = false,
                     bool overlap_at_start = true)
        : is_unidirectional_{exec->get_master(), num_blocks},
          overlaps_{exec->get_master(), num_blocks},
          overlap_at_start_{exec->get_master(), num_blocks}
    {
        // TODO move to a core function. and update to unidir and overlap_start
        // to have different values at start and end of arrays
        is_unidirectional_.fill(bool{is_unidirectional});
        is_unidirectional_.get_data()[0] = true;
        is_unidirectional_.get_data()[num_blocks - 1] = true;
        is_unidirectional_.set_executor(exec);

        overlap_at_start_.fill(bool{overlap_at_start});
        overlap_at_start_.get_data()[0] = false;
        overlap_at_start_.get_data()[num_blocks - 1] = true;
        overlap_at_start_.set_executor(exec);

        overlaps_.fill(ValueType{overlap});
    }

    template <typename OverlapArray, typename UnidirArray,
              typename StartEndArray>
    Overlap(std::shared_ptr<const Executor> exec, OverlapArray &&overlap,
            UnidirArray &&is_unidirectional, StartEndArray &&overlap_at_start)
        : is_unidirectional_{exec,
                             std::forward<UnidirArray>(is_unidirectional)},
          overlaps_{exec, std::forward<OverlapArray>(overlap)},
          overlap_at_start_{exec, std::forward<StartEndArray>(overlap_at_start)}
    {
        GKO_ASSERT(is_unidirectional_.get_num_elems() ==
                   overlaps_.get_num_elems());
        GKO_ASSERT(overlap_at_start_.get_num_elems() ==
                   overlaps_.get_num_elems());
    }

    Overlap(std::shared_ptr<const Executor> exec, const Overlap &other)
        : Overlap(exec)
    {
        *this = other;
    }

    Overlap(const Overlap &other) : Overlap(other.get_executor(), other) {}

    Overlap(std::shared_ptr<const Executor> exec, Overlap &&other)
        : Overlap(exec)
    {
        *this = std::move(other);
    }

    Overlap(Overlap &&other) : Overlap(other.get_executor(), std::move(other))
    {}

    Overlap &operator=(const Overlap &other)
    {
        if (&other == this) {
            return *this;
        }
        if (get_executor() == nullptr) {
            this->clear();
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        this->is_unidirectional_ = other.is_unidirectional_;
        this->overlaps_ = other.overlaps_;
        this->overlap_at_start_ = other.overlap_at_start_;
        return *this;
    }

    Overlap &operator=(Overlap &&other)
    {
        if (&other == this) {
            return *this;
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }
        if (get_executor() == other.get_executor()) {
            // same device, only move the pointer
            this->is_unidirectional_ = std::move(other.is_unidirectional_);
            this->overlaps_ = std::move(other.overlaps_);
            this->overlap_at_start_ = std::move(other.overlap_at_start_);
        } else {
            // different device, copy the data
            this->is_unidirectional_ = other.is_unidirectional_;
            this->overlaps_ = other.overlaps_;
            this->overlap_at_start_ = other.overlap_at_start_;
            *this = other;
        }
        return *this;
    }

    void clear() noexcept
    {
        this->is_unidirectional_.clear();
        this->overlaps_.clear();
        this->overlap_at_start_.clear();
    }

private:
    Array<bool> is_unidirectional_;
    Array<ValueType> overlaps_;
    Array<bool> overlap_at_start_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_OVERLAP_HPP_
