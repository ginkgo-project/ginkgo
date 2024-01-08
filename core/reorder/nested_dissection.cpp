// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/nested_dissection.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#if GKO_HAVE_METIS
#include GKO_METIS_HEADER
#endif


#include "core/base/allocator.hpp"


namespace gko {
namespace experimental {
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
        return "<" + std::to_string(metis_error) + ">";
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
    if (options.find(METIS_OPTION_NUMBERING) != options.end() &&
        options.at(METIS_OPTION_NUMBERING) != 0) {
        throw MetisError(
            __FILE__, __LINE__, "build_metis_options",
            "METIS_OPTION_NUMBERING: Only 0-based indexing is supported");
    }
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
    metis_nd<idx_t>(host_exec, num_rows, tmp_row_ptrs.data(),
                    tmp_col_idxs.data(), options, tmp_perm.data(),
                    tmp_iperm.data());
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
    std::shared_ptr<const LinOp> system_matrix) const
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
    exec->run(make_metis_nd(host_exec, num_rows, host_mtx->get_const_row_ptrs(),
                            host_mtx->get_const_col_idxs(),
                            build_metis_options(parameters_.options),
                            permutation.get_data(),
                            inv_permutation.get_data()));
    permutation.set_executor(exec);
    // we discard the inverse permutation
    return permutation_type::create(exec, std::move(permutation));
}


#define GKO_DECLARE_ND(ValueType, IndexType) \
    class NestedDissection<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ND);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
