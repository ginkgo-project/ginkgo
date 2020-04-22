/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/multigrid/amgx_pgm_kernels.hpp"


#include <memory>


#include <hip/hip_runtime.h>
#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/solver/common_trs_kernels.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The AMGX_PGM solver namespace.
 *
 * @ingroup amgx_pgm
 */
namespace amgx_pgm {


template <typename ValueType, typename IndexType>
void restrict_apply(std::shared_ptr<const HipExecutor> exec,
                    const Array<IndexType> &agg,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_RESTRICT_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void prolongate_applyadd(std::shared_ptr<const HipExecutor> exec,
                         const Array<IndexType> &agg,
                         const matrix::Dense<ValueType> *b,
                         matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_PROLONGATE_APPLY_KERNEL);


template <typename IndexType>
void initial(std::shared_ptr<const HipExecutor> exec,
             Array<IndexType> &agg) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_INITIAL_KERNEL);


template <typename IndexType>
void match_edge(std::shared_ptr<const HipExecutor> exec,
                const Array<IndexType> &strongest_neighbor,
                Array<IndexType> &agg) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const HipExecutor> exec,
                 const Array<IndexType> &agg,
                 size_type *num_unagg) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const HipExecutor> exec, Array<IndexType> &agg,
              size_type *num_agg) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diag(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Csr<ValueType, IndexType> *source,
                  Array<ValueType> &diag)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_EXTRACT_DIAG);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(std::shared_ptr<const HipExecutor> exec,
                             const matrix::Csr<ValueType, IndexType> *source,
                             const Array<ValueType> &diag,
                             Array<IndexType> &agg,
                             Array<IndexType> &strongest_neighbor)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR);


template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const HipExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *source,
                         const Array<ValueType> &diag, Array<IndexType> &agg,
                         Array<IndexType> &intermediate_agg)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG);


template <typename ValueType, typename IndexType>
void amgx_pgm_generate(std::shared_ptr<const HipExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       const Array<IndexType> &agg,
                       matrix::Csr<ValueType, IndexType> *coarse)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_GENERATE);


}  // namespace amgx_pgm
}  // namespace hip
}  // namespace kernels
}  // namespace gko
