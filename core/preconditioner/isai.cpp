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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <functional>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/preconditioner/isai_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace isai {


GKO_REGISTER_OPERATION(generate_tri_inverse, isai::generate_tri_inverse);
GKO_REGISTER_OPERATION(generate_excess_system, isai::generate_excess_system);
GKO_REGISTER_OPERATION(scatter_excess_solution, isai::scatter_excess_solution);


}  // namespace isai


/**
 * @internal
 *
 * Helper function that converts the given matrix to the (const) CSR format with
 * additional sorting.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `const Csr`, the same pointer is returned with an empty
 * deleter.
 * In all other cases, a new matrix is created, which stores the converted Csr
 * matrix.
 * If `skip_sorting` is false, the matrix will be sorted by column index,
 * otherwise, it will not be sorted.
 */
template <typename Csr>
std::shared_ptr<const Csr> convert_to_csr_and_sort(
    std::shared_ptr<const Executor> &exec, std::shared_ptr<const LinOp> mtx,
    bool skip_sorting)
{
    static_assert(
        std::is_same<Csr, matrix::Csr<typename Csr::value_type,
                                      typename Csr::index_type>>::value,
        "The given `Csr` type must be of type `matrix::Csr`!");
    if (skip_sorting && exec == mtx->get_executor()) {
        auto csr_mtx = std::dynamic_pointer_cast<const Csr>(mtx);
        if (csr_mtx) {
            // Here, we can just forward the pointer with an empty deleter
            // since it is already sorted and in the correct format
            return csr_mtx;
        }
    }
    auto copy = Csr::create(exec);
    as<ConvertibleTo<Csr>>(mtx)->convert_to(lend(copy));
    // Here, we assume that a sorted matrix converted to CSR will also be
    // sorted
    if (!skip_sorting) {
        copy->sort_by_column_index();
    }
    return {std::move(copy)};
}


/**
 * @internal
 *
 * Helper function that extends the sparsity pattern of the matrix M to M^n
 * without changing its values.
 *
 * The input matrix must be sorted and on the correct executor for this to work.
 * If `power` is 1, the matrix will be returned unchanged.
 */
template <typename Csr>
std::shared_ptr<Csr> extend_sparsity(std::shared_ptr<const Executor> &exec,
                                     std::shared_ptr<const Csr> mtx, int power)
{
    GKO_ASSERT_EQ(power >= 1, true);
    if (power == 1) {
        // copy the matrix, as it will be used to store the inverse
        return {std::move(mtx->clone())};
    }
    auto id_power = mtx->clone();
    auto tmp = Csr::create(exec, mtx->get_size());
    // accumulates mtx * the remainder from odd powers
    auto acc = mtx->clone();
    // compute id^(n-1) using square-and-multiply
    int i = power - 1;
    while (i > 1) {
        if (i % 2 != 0) {
            // store one power in acc:
            // i^(2n+1) -> i*i^2n
            id_power->apply(lend(acc), lend(tmp));
            std::swap(acc, tmp);
            i--;
        }
        // square id_power: i^2n -> (i^2)^n
        id_power->apply(lend(id_power), lend(tmp));
        std::swap(id_power, tmp);
        i /= 2;
    }
    // combine acc and id_power again
    id_power->apply(lend(acc), lend(tmp));
    return {std::move(tmp)};
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
void Isai<IsaiType, ValueType, IndexType>::generate_inverse(
    std::shared_ptr<const LinOp> input, bool skip_sorting, int power)
{
    using Dense = matrix::Dense<ValueType>;
    using LowerTrs = solver::LowerTrs<ValueType, IndexType>;
    using UpperTrs = solver::UpperTrs<ValueType, IndexType>;
    GKO_ASSERT_IS_SQUARE_MATRIX(input);
    auto exec = this->get_executor();
    auto to_invert = convert_to_csr_and_sort<Csr>(exec, input, skip_sorting);
    auto inverted = extend_sparsity(exec, to_invert, power);
    auto num_rows = inverted->get_size()[0];
    auto is_lower = IsaiType == isai_type::lower;

    // This stores the beginning of the RHS for the sparse block associated with
    // each row of inverted_l
    Array<IndexType> excess_block_ptrs{exec, num_rows + 1};
    // This stores the beginning of the non-zeros belonging to each row in the
    // system of excess blocks
    Array<IndexType> excess_row_ptrs_full{exec, num_rows + 1};

    exec->run(isai::make_generate_tri_inverse(
        lend(to_invert), lend(inverted), excess_block_ptrs.get_data(),
        excess_row_ptrs_full.get_data(), is_lower));

    auto excess_dim =
        exec->copy_val_to_host(excess_block_ptrs.get_const_data() + num_rows);
    // if we had long rows:
    if (excess_dim > 0) {
        // build the excess sparse triangular system
        auto excess_nnz = exec->copy_val_to_host(
            excess_row_ptrs_full.get_const_data() + num_rows);
        auto excess_system =
            Csr::create(exec, dim<2>(excess_dim, excess_dim), excess_nnz);
        auto excess_rhs = Dense::create(exec, dim<2>(excess_dim, 1));
        auto excess_solution = Dense::create(exec, dim<2>(excess_dim, 1));
        exec->run(isai::make_generate_excess_system(
            lend(to_invert), lend(inverted), excess_block_ptrs.get_const_data(),
            excess_row_ptrs_full.get_const_data(), lend(excess_system),
            lend(excess_rhs)));
        // solve it after transposing
        std::unique_ptr<LinOpFactory> trs_factory;
        if (is_lower) {
            trs_factory = UpperTrs::build().on(exec);
        } else {
            trs_factory = LowerTrs::build().on(exec);
        }
        trs_factory->generate(share(excess_system->transpose()))
            ->apply(lend(excess_rhs), lend(excess_solution));
        // and copy the results back to the original ISAI
        exec->run(isai::make_scatter_excess_solution(
            excess_block_ptrs.get_const_data(), lend(excess_solution),
            lend(inverted)));
    }

    approximate_inverse_ = std::move(inverted);
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Isai<IsaiType, ValueType, IndexType>::transpose() const
{
    std::unique_ptr<transposed_type> transp{
        new transposed_type{this->get_executor()}};
    transp->set_size(gko::transpose(this->get_size()));
    transp->approximate_inverse_ =
        share(as<Csr>(this->get_approximate_inverse()->transpose()));
    return transp;
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Isai<IsaiType, ValueType, IndexType>::conj_transpose()
    const
{
    std::unique_ptr<transposed_type> transp{
        new transposed_type{this->get_executor()}};
    transp->set_size(gko::transpose(this->get_size()));
    transp->approximate_inverse_ =
        share(as<Csr>(this->get_approximate_inverse()->conj_transpose()));
    return transp;
}


#define GKO_DECLARE_LOWER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::lower, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_ISAI);

#define GKO_DECLARE_UPPER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::upper, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_ISAI);


}  // namespace preconditioner
}  // namespace gko
