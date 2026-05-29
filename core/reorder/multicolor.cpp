// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/reorder/multicolor_kernels.hpp"


namespace gko {
namespace reorder {
namespace multicolor {
namespace {


GKO_REGISTER_OPERATION(compute_permutation_csr,
                       multicolor::compute_permutation_csr);


}  // anonymous namespace
}  // namespace multicolor


template <template <typename, typename> class MatrixTempl, typename ValueType,
          typename IndexType>
void multicolor_reorder(const MatrixTempl<ValueType, IndexType>* const mtx,
                        gko::array<IndexType>& color_ptrs,
                        IndexType* const permutation,
                        IndexType* const inv_permutation)
{
    const auto exec = mtx->get_executor();
    const IndexType num_rows = mtx->get_size()[0];
    exec->run(multicolor::make_compute_permutation_csr(
        num_rows, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        color_ptrs, permutation, inv_permutation));
}


template <typename ValueType, typename IndexType>
Multicolor<ValueType, IndexType>::Multicolor(
    std::shared_ptr<const Executor> exec)
    : EnablePolymorphicObject<Multicolor, ReorderingBase<IndexType>>(
          std::move(exec)),
      color_ptrs_{this->get_executor()->get_master()}
{}


template <typename ValueType, typename IndexType>
Multicolor<ValueType, IndexType>::Multicolor(const Factory* factory,
                                             const ReorderingBaseArgs& args)
    : EnablePolymorphicObject<Multicolor, ReorderingBase<IndexType>>(
          factory->get_executor()),
      parameters_{factory->get_parameters()},
      color_ptrs_{factory->get_executor()->get_master()}
{
    using CsrType = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto sysmat = args.system_matrix;
    auto const size = sysmat->get_size()[0];

    std::shared_ptr<const SparsityMatrix> matrix;

    if (auto csrmat = std::dynamic_pointer_cast<const CsrType>(sysmat)) {
        const auto n = csrmat->get_size()[0];
        const auto nnz = csrmat->get_num_stored_elements();
        matrix = SparsityMatrix::create_const(
            exec, sysmat->get_size(),
            make_const_array_view<IndexType>(exec, nnz,
                                             csrmat->get_const_col_idxs()),
            make_const_array_view<IndexType>(exec, size + 1,
                                             csrmat->get_const_row_ptrs()));
    } else if (auto smat =
                   std::dynamic_pointer_cast<const SparsityMatrix>(sysmat)) {
        matrix = smat;
    } else {
        GKO_NOT_SUPPORTED(sysmat);
    }

    // The adjacency matrix has to be square.
    GKO_ASSERT_IS_SQUARE_MATRIX(sysmat);

    permutation_ = PermutationMatrix::create(exec, size);
    inv_permutation_ = PermutationMatrix::create(exec, size);

    multicolor_reorder(matrix.get(), color_ptrs_,
                       permutation_->get_permutation(),
                       inv_permutation_->get_permutation());
}


#define GKO_DECLARE_MULTICOLOR(ValueType, IndexType) \
    class Multicolor<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MULTICOLOR);


}  // namespace reorder
}  // namespace gko
