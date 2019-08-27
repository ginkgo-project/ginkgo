/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/factorization/qr.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/perturbation.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>


#include "core/factorization/qr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace qr_factorization {


GKO_REGISTER_OPERATION(householder_generator,
                       qr_factorization::householder_generator);


}  // namespace qr_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Qr<ValueType>::generate_qr(
    const std::shared_ptr<const LinOp> &system_matrix) const
{
    using DenseMatrix = matrix::Dense<ValueType>;

    const auto exec = this->get_executor();

    // Only copies the matrix if it is not on the same executor or was not in
    // the right format. Throws an exception if it is not convertable.
    std::unique_ptr<DenseMatrix> dense_system_matrix_unique_ptr{};
    auto dense_system_matrix =
        dynamic_cast<const DenseMatrix *>(system_matrix.get());
    if (dense_system_matrix == nullptr ||
        dense_system_matrix->get_executor() != exec) {
        dense_system_matrix_unique_ptr = DenseMatrix::create(exec);
        as<ConvertibleTo<DenseMatrix>>(system_matrix.get())
            ->convert_to(dense_system_matrix_unique_ptr.get());
        dense_system_matrix = dense_system_matrix_unique_ptr.get();
    }
    // QR algorithm changes the value of the matrix, so it needs a copy of
    // system_matrix.
    auto work_matrix = dense_system_matrix.clone();
    const auto nrows = work_matrix->get_size()[0];
    const auto ncols = work_matrix->get_size()[1];
    const auto rank = (parameters_.rank == 0) ? ncols : parameters_.rank;
    auto std::vector<std::unique_ptr<Perturbation<ValueType>>> q_list(rank);
    for (size_type i = 0; i < rank; i++) {
        // householder_generator(vector, k) -> factor, scalar
        factor = DenseMatrix::create(exec, dim<2>{nrows, 1});
        scalar = initialize<DenseMatrix>({gko::one<ValueType>()}, exec);
        x = work_matrix->create_submatrix(span{0, nrows}, span{i, i + 1});
        exec->run(
            qr_factorization::make_householder_generator(x, i, lend(factor)));
        // apply on submatrix of work_matrix
        auto sub_factor = factor->create_submatrix(span{i, nrows}, span{0, 1});
        auto sub_work_matrix =
            work_matrix->create_submatrix(span{i, nrows}, span{i, ncols});
        auto sub_householder_matrix =
            Perturbation<ValueType>::create(scalar, sub_factor);
        auto temp_matrix = clone(sub_work_matrix);
        sub_householder_matrix->apply(sub_work_matrix, temp_matrix);
        sub_work_matrix->copy_from(temp_matrix);
        // create a perturbation to present housholder matrix
        auto householder_matrix =
            Perturbation<ValueType>::create(scalar, factor);
        q_list.push_back(std::move(householder_matrix));
    }
    auto q = Composition<ValueType>::create(q_list.cbegin(), q_list.cend());
    auto r = work_matrix->create_submatrix(span{0, rank}, span{0, ncols});
    return Composition<ValueType>::create(std::move(q), std::move(r));
}


#define GKO_DECLARE_QR(ValueType) class Qr<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_QR);


}  // namespace factorization
}  // namespace gko
