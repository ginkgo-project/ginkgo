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

#ifndef GKO_PUBLIC_CORE_MULTIGRID_MAPPING_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_MAPPING_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_base.hpp>


namespace gko {
namespace multigrid {


template <typename ValueType, typename IndexType>
class Mapping : public EnableLinOp<Mapping<ValueType, IndexType>>,
                public EnableCreateMethod<Mapping<ValueType, IndexType>> {
    friend class EnableCreateMethod<Mapping>;
    friend class EnablePolymorphicObject<Mapping, LinOp>;

public:
    using index_type = IndexType;

protected:
    /**
     * Creates an uninitialized Mapping arrays on the specified executor..
     *
     * @param exec  Executor associated to the LinOp
     */
    Mapping(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Mapping>(exec), mapping_{exec}, is_restrict_(true)
    {}

    template <typename IndicesArray>
    Mapping(std::shared_ptr<const Executor> exec, const dim<2> &size,
            IndicesArray &&mapping_indices, const bool &is_restrict = true)
        : EnableLinOp<Mapping>(exec, size),
          mapping_{exec, std::forward<IndicesArray>(mapping_indices)},
          is_restrict_(is_restrict)
    {
        if (is_restrict_) {
            GKO_ASSERT_EQ(size[1], mapping_.get_num_elems());
        }
        if (!is_restrict_) {
            GKO_ASSERT_EQ(size[0], mapping_.get_num_elems());
        }
    }

    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

private:
    Array<index_type> mapping_;
    bool is_restrict_;
};


}  // namespace multigrid
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_MULTIGRID_MAPPING_HPP_