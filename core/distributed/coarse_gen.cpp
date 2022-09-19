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

#include <ginkgo/core/distributed/coarse_gen.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/coarse_gen_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/amgx_pgm_kernels.hpp"


namespace gko {
namespace distributed {
namespace coarse_gen {
namespace {


GKO_REGISTER_OPERATION(match_edge, amgx_pgm::match_edge);
GKO_REGISTER_OPERATION(count_unagg, amgx_pgm::count_unagg);
GKO_REGISTER_OPERATION(renumber, amgx_pgm::renumber);
GKO_REGISTER_OPERATION(find_strongest_neighbor,
                       coarse_gen::find_strongest_neighbor);
GKO_REGISTER_OPERATION(assign_to_exist_agg, coarse_gen::assign_to_exist_agg);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace coarse_gen


template <typename ValueType, typename IndexType>
void CoarseGen<ValueType, IndexType>::generate_with_aggregation()
{
    GKO_NOT_IMPLEMENTED;
}


template <typename ValueType, typename IndexType>
void CoarseGen<ValueType, IndexType>::generate_with_selection()
{
    using matrix_type = distributed::Matrix<ValueType, IndexType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_matrix_type = remove_complex<matrix_type>;
    auto exec = this->get_executor();
    const matrix_type* dist_mat =
        dynamic_cast<const matrix_type*>(system_matrix_.get());

    const auto global_num_rows = dist_mat->get_size()[0];
    const auto local_num_rows = dist_mat->get_local_matrix()->get_size()[0];
}


#define GKO_DECLARE_COARSE_GEN(_vtype, _itype) class CoarseGen<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSE_GEN);


}  // namespace distributed
}  // namespace gko
