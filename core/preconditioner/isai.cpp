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


#include "core/preconditioner/isai_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace isai {


GKO_REGISTER_OPERATION(identity_triangle, isai::identity_triangle);
GKO_REGISTER_OPERATION(generate_l_inverse, isai::generate_l_inverse);
GKO_REGISTER_OPERATION(generate_u_inverse, isai::generate_u_inverse);


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
    as<ConvertibleTo<Csr>>(mtx)->convert_to(copy.get());
    // Here, we assume that a sorted matrix converted to CSR will also be
    // sorted
    if (!skip_sorting) {
        copy->sort_by_column_index();
    }
    return copy;
}


/**
 * @internal
 *
 * Helper function that extends the sparsity pattern of the matrix M to M^n
 * without changing its values.
 *
 * The input matrix must be sorted and on the correct executor for this to work.
 * If `power` is 1, the matrix will be returned unchanged. *
 */
template <typename Csr>
std::shared_ptr<const Csr> extend_sparsity(
    std::shared_ptr<const Executor> &exec, std::shared_ptr<const Csr> mtx,
    int power, bool lower)
{
    GKO_ASSERT_EQ(power >= 1, true);
    if (power == 1) {
        return mtx;
    }
    auto id = mtx->clone();
    exec->run(isai::make_identity_triangle(id.get(), lower));
    auto id_power = id->clone();
    auto tmp = Csr::create(exec, mtx->get_size());
    // compute id^(n-1) and then multiply it with mtx
    // TODO replace this by a square-and-multiply algorithm
    for (int i = 1; i < power - 1; ++i) {
        id->apply(id_power.get(), tmp.get());
        std::swap(id_power, tmp);
    }
    // finally compute id^(n-1) * mtx
    id_power->apply(mtx.get(), tmp.get());
    return tmp;
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
void Isai<IsaiType, ValueType, IndexType>::generate_l_inverse(
    std::shared_ptr<const LinOp> to_invert_l, bool skip_sorting, int power)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(to_invert_l);
    auto exec = this->get_executor();
    auto csr_l = convert_to_csr_and_sort<Csr>(exec, to_invert_l, skip_sorting);
    auto strategy = csr_l->get_strategy();
    auto csr_extended_l = extend_sparsity(exec, csr_l, power, true);
    const auto num_elems = csr_extended_l->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_l =
        Csr::create(exec, csr_extended_l->get_size(), num_elems, strategy);
    exec->run(
        isai::make_generate_l_inverse(csr_extended_l.get(), inverted_l.get()));

    approximate_inverse_ = std::move(inverted_l);
}


template <isai_type IsaiType, typename ValueType, typename IndexType>
void Isai<IsaiType, ValueType, IndexType>::generate_u_inverse(
    std::shared_ptr<const LinOp> to_invert_u, bool skip_sorting, int power)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(to_invert_u);
    auto exec = this->get_executor();
    auto csr_u = convert_to_csr_and_sort<Csr>(exec, to_invert_u, skip_sorting);
    auto strategy = csr_u->get_strategy();
    auto csr_extended_u = extend_sparsity(exec, csr_u, power, false);
    const auto num_elems = csr_extended_u->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_u =
        Csr::create(exec, csr_extended_u->get_size(), num_elems, strategy);
    exec->run(
        isai::make_generate_u_inverse(csr_extended_u.get(), inverted_u.get()));

    approximate_inverse_ = std::move(inverted_u);
}


#define GKO_DECLARE_LOWER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::lower, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_ISAI);

#define GKO_DECLARE_UPPER_ISAI(ValueType, IndexType) \
    class Isai<isai_type::upper, ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_ISAI);


}  // namespace preconditioner
}  // namespace gko
