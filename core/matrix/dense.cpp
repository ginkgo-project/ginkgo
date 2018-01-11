#include "core/matrix/dense.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply<ValueType>);
    GKO_REGISTER_OPERATION(apply, dense::apply<ValueType>);
    GKO_REGISTER_OPERATION(scale, dense::scale<ValueType>);
    GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled<ValueType>);
    GKO_REGISTER_OPERATION(compute_dot, dense::compute_dot<ValueType>);
};


}  // namespace


template <typename ValueType>
void Dense<ValueType>::copy_from(const LinOp *other)
{
    as<ConvertibleTo<Dense<ValueType>>>(other)->convert_to(this);
}


template <typename ValueType>
void Dense<ValueType>::copy_from(std::unique_ptr<LinOp> other)
{
    as<ConvertibleTo<Dense<ValueType>>>(other.get())->move_to(this);
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();
    if (b->get_executor() != exec || x->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_simple_apply_operation(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                             const LinOp *beta, LinOp *x) const
{
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec || b->get_executor() != exec ||
        beta->get_executor() != exec || x->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_apply_operation(
        as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
        as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::scale(const LinOp *alpha)
{
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_scale_operation(
        as<Dense<ValueType>>(alpha), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(const LinOp *alpha, const LinOp *b)
{
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec || b->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_add_scaled_operation(
        as<Dense<ValueType>>(alpha), as<Dense<ValueType>>(b), this));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(const LinOp *b, LinOp *result) const
{
    auto exec = this->get_executor();
    if (b->get_executor() != exec || result->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_compute_dot_operation(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(result)));
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
    result->set_dimensions(this);
    result->values_ = values_;
    result->padding_ = padding_;
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense *result)
{
    result->set_dimensions(this);
    result->values_ = std::move(values_);
    result->padding_ = padding_;
}


#define DECLARE_DENSE_MATRIX(_type) class Dense<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
