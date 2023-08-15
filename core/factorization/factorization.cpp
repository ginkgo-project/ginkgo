// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/factorization.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace experimental {
namespace factorization {


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::unpack() const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
storage_type Factorization<ValueType, IndexType>::get_storage_type() const
{
    return storage_type_;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>>
Factorization<ValueType, IndexType>::get_lower_factor() const
{
    switch (storage_type_) {
    case storage_type::composition:
    case storage_type::symm_composition:
        GKO_ASSERT(factors_->get_operators().size() == 2 ||
                   factors_->get_operators().size() == 3);
        return as<matrix_type>(factors_->get_operators()[0]);
    case storage_type::empty:
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
    default:
        return nullptr;
    }
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const gko::matrix::Diagonal<ValueType>>
Factorization<ValueType, IndexType>::get_diagonal() const
{
    switch (storage_type_) {
    case storage_type::composition:
    case storage_type::symm_composition:
        if (factors_->get_operators().size() == 3) {
            return as<diag_type>(factors_->get_operators()[1]);
        } else {
            return nullptr;
        }
    case storage_type::empty:
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
    default:
        return nullptr;
    }
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>>
Factorization<ValueType, IndexType>::get_upper_factor() const
{
    switch (storage_type_) {
    case storage_type::composition:
    case storage_type::symm_composition:
        GKO_ASSERT(factors_->get_operators().size() == 2 ||
                   factors_->get_operators().size() == 3);
        return as<matrix_type>(factors_->get_operators().back());
    case storage_type::empty:
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
    default:
        return nullptr;
    }
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>>
Factorization<ValueType, IndexType>::get_combined() const
{
    switch (storage_type_) {
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
        GKO_ASSERT(factors_->get_operators().size() == 1);
        return as<matrix_type>(factors_->get_operators()[0]);
    case storage_type::empty:
    case storage_type::composition:
    case storage_type::symm_composition:
    default:
        return nullptr;
    }
}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>::Factorization(const Factorization& fact)
    : Factorization{fact.get_executor()}
{
    *this = fact;
}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>::Factorization(Factorization&& fact)
    : Factorization{fact.get_executor()}
{
    *this = std::move(fact);
}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>&
Factorization<ValueType, IndexType>::operator=(const Factorization& fact)
{
    if (this != &fact) {
        EnableLinOp<Factorization<ValueType, IndexType>>::operator=(fact);
        storage_type_ = fact.storage_type_;
        *factors_ = *fact.factors_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>&
Factorization<ValueType, IndexType>::operator=(Factorization&& fact)
{
    if (this != &fact) {
        EnableLinOp<Factorization<ValueType, IndexType>>::operator=(
            std::move(fact));
        storage_type_ = std::exchange(fact.storage_type_, storage_type::empty);
        factors_ =
            std::exchange(fact.factors_, fact.factors_->create_default());
        if (factors_->get_executor() != this->get_executor()) {
            factors_ = factors_->clone(this->get_executor());
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>::Factorization(
    std::shared_ptr<const Executor> exec)
    : EnableLinOp<Factorization<ValueType, IndexType>>{exec},
      storage_type_{storage_type::empty},
      factors_{Composition<ValueType>::create(exec)}
{}


template <typename ValueType, typename IndexType>
Factorization<ValueType, IndexType>::Factorization(
    std::unique_ptr<Composition<ValueType>> factors, storage_type type)
    : EnableLinOp<Factorization<ValueType, IndexType>>{factors->get_executor(),
                                                       factors->get_size()},
      storage_type_{type},
      factors_{std::move(factors)}
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_composition(
    std::unique_ptr<composition_type> composition)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{std::move(composition),
                                                storage_type::composition}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_symm_composition(
    std::unique_ptr<composition_type> composition)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{
            std::move(composition), storage_type::symm_composition}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_combined_lu(
    std::unique_ptr<matrix_type> combined)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{
            composition_type::create(gko::share(std::move(combined))),
            storage_type::combined_lu}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_combined_ldu(
    std::unique_ptr<matrix_type> combined)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{
            composition_type::create(gko::share(std::move(combined))),
            storage_type::combined_ldu}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_combined_cholesky(
    std::unique_ptr<matrix_type> combined)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{
            composition_type::create(gko::share(std::move(combined))),
            storage_type::symm_combined_cholesky}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Factorization<ValueType, IndexType>::create_from_combined_ldl(
    std::unique_ptr<matrix_type> combined)
{
    return std::unique_ptr<Factorization<ValueType, IndexType>>{
        new Factorization<ValueType, IndexType>{
            composition_type::create(gko::share(std::move(combined))),
            storage_type::symm_combined_ldl}};
}


template <typename ValueType, typename IndexType>
void Factorization<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                     LinOp* x) const
{
    switch (storage_type_) {
    case storage_type::composition:
    case storage_type::symm_composition:
        factors_->apply(b, x);
        break;
    case storage_type::empty:
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
    default:
        GKO_NOT_SUPPORTED(storage_type_);
    }
}


template <typename ValueType, typename IndexType>
void Factorization<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                     const LinOp* b,
                                                     const LinOp* beta,
                                                     LinOp* x) const
{
    switch (storage_type_) {
    case storage_type::composition:
    case storage_type::symm_composition:
        factors_->apply(alpha, b, beta, x);
        break;
    case storage_type::empty:
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
    default:
        GKO_NOT_SUPPORTED(storage_type_);
    }
}


#define GKO_DECLARE_FACTORIZATION(ValueType, IndexType) \
    class Factorization<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FACTORIZATION);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
