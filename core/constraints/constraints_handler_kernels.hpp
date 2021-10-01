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

#ifndef GKO_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_KERNELS_HPP_
#define GKO_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_KERNELS_HPP_


#include <ginkgo/core/constraints/constraints_handler.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CONS_FILL_SUBSET(ValueType, IndexType)            \
    void fill_subset(std::shared_ptr<const DefaultExecutor> exec,     \
                     const Array<IndexType>& subset, ValueType* data, \
                     ValueType val)


#define GKO_DECLARE_CONS_COPY_SUBSET(ValueType, IndexType)                 \
    void copy_subset(std::shared_ptr<const DefaultExecutor> exec,          \
                     const Array<IndexType>& subset, const ValueType* src, \
                     ValueType* dst)


#define GKO_DECLARE_CONS_SET_UNIT_ROWS(ValueType, IndexType)                 \
    void set_unit_rows(std::shared_ptr<const DefaultExecutor> exec,          \
                       const Array<IndexType>& subset,                       \
                       const IndexType* row_ptrs, const IndexType* col_idxs, \
                       ValueType* values)

#define GKO_DECLARE_ALL_AS_TEMPLATES                    \
    template <typename ValueType, typename IndexType>   \
    GKO_DECLARE_CONS_FILL_SUBSET(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>   \
    GKO_DECLARE_CONS_COPY_SUBSET(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>   \
    GKO_DECLARE_CONS_SET_UNIT_ROWS(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cons, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_KERNELS_HPP_
