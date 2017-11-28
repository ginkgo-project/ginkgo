#include "core/matrix/dense.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename ValueType>
struct TemplatedGemmOperation {
    GKO_REGISTER_OPERATION(gemm, gemm<ValueType>);
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
    this->apply(1.0, b, 0.0, x);
}


template <typename ValueType>
void Dense<ValueType>::apply(full_precision alpha, const LinOp *b,
                             full_precision beta, LinOp *x) const
{
    auto dense_b = dynamic_cast<const Dense *>(b);
    auto dense_x = dynamic_cast<Dense *>(x);
    if (dense_b == nullptr || dense_b->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(b);
    }
    if (dense_x == nullptr || dense_x->get_executor() != this->get_executor()) {
        throw NOT_SUPPORTED(x);
    }

    this->get_executor()->run(
        TemplatedGemmOperation<ValueType>::make_gemm_operation(
            static_cast<ValueType>(alpha), this, dense_b,
            static_cast<ValueType>(beta), dense_x));
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
