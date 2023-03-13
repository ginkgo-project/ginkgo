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

#include <ginkgo/core/reorder/nested_dissection.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#if GKO_HAVE_METIS
#include GKO_METIS_HEADER
#endif


#include <ginkgo/core/base/temporary_clone.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace reorder {
namespace {


std::string metis_error_message(idx_t metis_error)
{
    switch (metis_error) {
    case METIS_ERROR_INPUT:
        return "METIS_ERROR_INPUT";
    case METIS_ERROR_MEMORY:
        return "METIS_ERROR_MEMORY";
    case METIS_ERROR:
        return "METIS_ERROR";
    case METIS_OK:
        return "METIS_OK";
    default:
        return "<unknown>";
    }
}


std::array<idx_t, METIS_NOPTIONS> build_metis_options(
    const std::unordered_map<int, int>& options)
{
    std::array<idx_t, METIS_NOPTIONS> result{};
    METIS_SetDefaultOptions(result.data());
    for (auto pair : options) {
        if (pair.first < 0 || pair.first >= METIS_NOPTIONS) {
            throw MetisError(__FILE__, __LINE__, "build_metis_options",
                             "Invalid option ID " + std::to_string(pair.first));
        }
        result[pair.first] = pair.second;
    }
    // make sure users don't accidentally switch on 1-based indexing
    result[METIS_OPTION_NUMBERING] = 0;
    return result;
}


template <typename IndexType>
void metis_nd(std::shared_ptr<const Executor> host_exec, size_type num_rows,
              const IndexType* row_ptrs, const IndexType* col_idxs,
              const std::array<idx_t, METIS_NOPTIONS>& options, IndexType* perm,
              IndexType* iperm)
{
    auto nvtxs = static_cast<idx_t>(num_rows);
    vector<idx_t> tmp_row_ptrs(row_ptrs, row_ptrs + num_rows + 1, {host_exec});
    vector<idx_t> tmp_col_idxs(col_idxs, col_idxs + row_ptrs[num_rows],
                               {host_exec});
    vector<idx_t> tmp_perm(num_rows, {host_exec});
    vector<idx_t> tmp_iperm(num_rows, {host_exec});
    auto result = METIS_NodeND(&nvtxs, tmp_row_ptrs.data(), tmp_col_idxs.data(),
                               nullptr, const_cast<idx_t*>(options.data()),
                               tmp_perm.data(), tmp_iperm.data());
    if (result != METIS_OK) {
        throw MetisError(__FILE__, __LINE__, "METIS_NodeND",
                         metis_error_message(result));
    }
    std::copy_n(tmp_perm.begin(), num_rows, perm);
    std::copy_n(tmp_iperm.begin(), num_rows, iperm);
}


template <>
void metis_nd<idx_t>(std::shared_ptr<const Executor> host_exec,
                     size_type num_rows, const idx_t* row_ptrs,
                     const idx_t* col_idxs,
                     const std::array<idx_t, METIS_NOPTIONS>& options,
                     idx_t* perm, idx_t* iperm)
{
    auto nvtxs = static_cast<idx_t>(num_rows);
    auto result = METIS_NodeND(&nvtxs, const_cast<idx_t*>(row_ptrs),
                               const_cast<idx_t*>(col_idxs), nullptr,
                               const_cast<idx_t*>(options.data()), perm, iperm);
    if (result != METIS_OK) {
        throw MetisError(__FILE__, __LINE__, "METIS_NodeND",
                         metis_error_message(result));
    }
}


GKO_REGISTER_HOST_OPERATION(metis_nd, metis_nd);


}  // namespace


template <typename ValueType, typename IndexType>
NestedDissection<ValueType, IndexType>::NestedDissection(
    std::shared_ptr<const Executor> exec, const parameters_type& params)
    : EnablePolymorphicObject<NestedDissection, LinOpFactory>(std::move(exec)),
      parameters_(params)
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Permutation<IndexType>>
NestedDissection<ValueType, IndexType>::generate(
    std::shared_ptr<const matrix_type> system_matrix) const
{
    auto product =
        std::unique_ptr<permutation_type>(static_cast<permutation_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> NestedDissection<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    // most matrix formats are convertible to Csr, but not to SparsityCsr, so we
    // take the detour through Csr
    const auto csr_mtx = copy_and_convert_to<matrix_type>(exec, system_matrix);
    const auto sparsity_mtx =
        copy_and_convert_to<matrix::SparsityCsr<ValueType, IndexType>>(exec,
                                                                       csr_mtx)
            ->to_adjacency_matrix();
    const auto host_mtx = make_temporary_clone(host_exec, sparsity_mtx);
    const auto num_rows = host_mtx->get_size()[0];
    array<IndexType> permutation(host_exec, num_rows);
    array<IndexType> inv_permutation(host_exec, num_rows);
    exec->run(make_metis_nd(exec, num_rows, host_mtx->get_const_row_ptrs(),
                            host_mtx->get_const_col_idxs(),
                            build_metis_options(parameters_.options),
                            permutation.get_data(),
                            inv_permutation.get_data()));
    permutation.set_executor(exec);
    // we discard the inverse permutation
    return permutation_type::create(exec, dim<2>{num_rows, num_rows},
                                    std::move(permutation));
}


#define GKO_DECLARE_ND(ValueType, IndexType) \
    class NestedDissection<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ND);


}  // namespace reorder
}  // namespace gko
