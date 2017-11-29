#include "core/matrix/dense.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(gemm, gemm<ValueType>);
    GKO_REGISTER_OPERATION(scale, scal<ValueType>);
    GKO_REGISTER_OPERATION(add_scaled, axpy<ValueType>);
    GKO_REGISTER_OPERATION(compute_dot, dot<ValueType>);
};


}  // namespace


template <typename ValueType>
void Dense<ValueType>::copy_from(const LinOp *other)
{
    auto convertible_to_dense =
        dynamic_cast<const ConvertibleTo<Dense<ValueType>> *>(other);
    if (convertible_to_dense != nullptr) {
        convertible_to_dense->convert_to(this);
    } else {
        throw NOT_SUPPORTED(other);
    }
}


template <typename ValueType>
void Dense<ValueType>::copy_from(std::unique_ptr<LinOp> other)
{
    auto convertible_to_dense =
        dynamic_cast<ConvertibleTo<Dense<ValueType>> *>(other.get());
    if (convertible_to_dense != nullptr) {
        convertible_to_dense->move_to(this);
    } else {
        throw NOT_SUPPORTED(other);
    }
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    // TODO: adding some of these constants to executors can potentially save a
    //       lot of runtime on memory allocations / deallocations
    auto zero = Dense::create(this->get_executor(), {ValueType(0)});
    auto one = Dense::create(this->get_executor(), {ValueType(1)});
    this->apply(one.get(), b, zero.get(), x);
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                             const LinOp *beta, LinOp *x) const
{
    auto dense_b = dynamic_cast<const Dense *>(b);
    auto dense_alpha = dynamic_cast<const Dense *>(alpha);
    auto dense_beta = dynamic_cast<const Dense *>(beta);
    auto dense_x = dynamic_cast<Dense *>(x);
    if (dense_b == nullptr || dense_b->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(b);
    }
    if (dense_alpha == nullptr ||
        dense_alpha->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(alpha);
    }
    if (dense_beta == nullptr ||
        dense_beta->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(beta);
    }
    if (dense_x == nullptr || dense_x->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(x);
    }

    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_gemm_operation(
            dense_alpha, this, dense_b, dense_beta, dense_x));
}


template <typename ValueType>
void Dense<ValueType>::scale(const LinOp *alpha)
{
    auto dense_alpha = dynamic_cast<const Dense *>(alpha);
    if (dense_alpha == nullptr ||
        dense_alpha->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(alpha);
    }
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_scale_operation(dense_alpha, this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(const LinOp *alpha, const LinOp *b)
{
    auto dense_alpha = dynamic_cast<const Dense *>(alpha);
    auto dense_b = dynamic_cast<const Dense *>(b);
    if (dense_alpha == nullptr ||
        dense_alpha->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(alpha);
    }
    if (dense_b == nullptr || dense_b->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(b);
    }
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_add_scaled_operation(
            dense_alpha, dense_b, this));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(const LinOp *b, LinOp *result) const
{
    auto dense_b = dynamic_cast<const Dense *>(b);
    auto dense_result = dynamic_cast<Dense *>(result);
    if (dense_b == nullptr || dense_b->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(b);
    }
    if (dense_result == nullptr ||
        dense_result->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(result);
    }
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_compute_dot_operation(
            this, dense_b, dense_result));
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::clone_type() const
{
    return std::unique_ptr<Dense>(new Dense(this->get_executor(), 0, 0, 0));
}


template <typename ValueType>
void Dense<ValueType>::clear()
{
    this->set_dimensions(0, 0, 0);
    values_.clear();
    padding_ = 0;
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Dense *result) const
{
    *result = *this;
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense *result)
{
    *result = std::move(*this);
}


#define DECLARE_DENSE_MATRIX(_type) class Dense<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
