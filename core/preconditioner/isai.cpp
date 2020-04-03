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


GKO_REGISTER_OPERATION(generate_l_inverse, isai::generate_l_inverse);
GKO_REGISTER_OPERATION(generate_u_inverse, isai::generate_u_inverse);


}  // namespace isai


/**
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
std::unique_ptr<const Csr, std::function<void(const Csr *)>>
convert_to_csr_and_sort(std::shared_ptr<const Executor> &exec, const LinOp *mtx,
                        bool skip_sorting)
{
    static_assert(
        std::is_same<Csr, matrix::Csr<typename Csr::value_type,
                                      typename Csr::index_type>>::value,
        "The given `Csr` type must be of type `matrix::Csr`!");
    if (skip_sorting && exec == mtx->get_executor()) {
        auto csr_mtx = dynamic_cast<const Csr *>(mtx);
        if (csr_mtx) {
            // Here, we can just forward the pointer with an empty deleter
            // since it is already sorted and in the correct format
            return {csr_mtx, [](const Csr *) {}};
        }
    }
    auto copy = Csr::create(exec);
    as<ConvertibleTo<Csr>>(mtx)->convert_to(copy.get());
    // Here, we assume that a sorted matrix converted to CSR will also be
    // sorted
    if (!skip_sorting) {
        copy->sort_by_column_index();
    }
    return {copy.release(), std::default_delete<const Csr>{}};
}


template <typename ValueType, typename IndexType>
void Isai<ValueType, IndexType>::generate_l_inverse(const LinOp *to_invert_l,
                                                    bool skip_sorting)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(to_invert_l);
    auto exec = this->get_executor();
    auto csr_l = convert_to_csr_and_sort<Csr>(exec, to_invert_l, skip_sorting);
    const auto num_elems = csr_l->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_l =
        Csr::create(exec, csr_l->get_size(), num_elems, csr_l->get_strategy());
    exec->run(isai::make_generate_l_inverse(csr_l.get(), inverted_l.get()));

    l_inv_ = std::move(inverted_l);
}


template <typename ValueType, typename IndexType>
void Isai<ValueType, IndexType>::generate_u_inverse(const LinOp *to_invert_u,
                                                    bool skip_sorting)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(to_invert_u);
    auto exec = this->get_executor();
    auto csr_u = convert_to_csr_and_sort<Csr>(exec, to_invert_u, skip_sorting);
    const auto num_elems = csr_u->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_u =
        Csr::create(exec, csr_u->get_size(), num_elems, csr_u->get_strategy());
    exec->run(isai::make_generate_u_inverse(csr_u.get(), inverted_u.get()));

    u_inv_ = std::move(inverted_u);
}


#define GKO_DECLARE_ISAI(ValueType, IndexType) class Isai<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI);


}  // namespace preconditioner
}  // namespace gko
