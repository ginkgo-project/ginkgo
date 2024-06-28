// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>


namespace gko {
namespace preconditioner {


template <typename ValueType, typename IndexType>
std::unique_ptr<typename GaussSeidel<ValueType, IndexType>::composition_type>
GaussSeidel<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<composition_type>(static_cast<composition_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    return Sor<ValueType, IndexType>::build()
        .with_skip_sorting(parameters_.skip_sorting)
        .with_symmetric(parameters_.symmetric)
        .with_relaxation_factor(static_cast<remove_complex<ValueType>>(1.0))
        .with_l_solver(parameters_.l_solver)
        .with_u_solver(parameters_.u_solver)
        .on(this->get_executor())
        ->generate(std::move(system_matrix));
}


#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);


}  // namespace preconditioner
}  // namespace gko
