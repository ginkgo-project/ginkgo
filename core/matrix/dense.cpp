#include "core/matrix/dense.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"


#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>


namespace gko {


namespace matrix {


namespace detail {


template <int K, int... Ns, typename F, typename Tuple>
typename std::enable_if<(K == 0)>::type call_impl(F f, Tuple &&data)
{
    f(std::get<Ns>(std::forward<Tuple>(data))...);
}

template <int K, int... Ns, typename F, typename Tuple>
typename std::enable_if<(K > 0)>::type call_impl(F f, Tuple &&data)
{
    call_impl<K - 1, K - 1, Ns...>(f, std::forward<Tuple>(data));
}

template <typename F, typename... Args>
void call(F f, const std::tuple<Args...> &data)
{
    call_impl<sizeof...(Args)>(f, data);
}


}  // namespace detail


#define GINKGO_REGISTER_OPERATION(_name, _kernel)                              \
    template <typename... Args>                                                \
    class _name##_operation : public Operation {                               \
    public:                                                                    \
        _name##_operation(Args... args) : data(std::forward<Args>(args)...) {} \
                                                                               \
        void run(const CpuExecutor *) const override                           \
        {                                                                      \
            detail::call(kernels::cpu::_kernel, data);                         \
        }                                                                      \
                                                                               \
        void run(const GpuExecutor *) const override                           \
        {                                                                      \
            detail::call(kernels::gpu::_kernel, data);                         \
        }                                                                      \
                                                                               \
    private:                                                                   \
        std::tuple<Args...> data;                                              \
    };                                                                         \
                                                                               \
    template <typename... Args>                                                \
    static _name##_operation<Args...> make_##_name##_operation(                \
        Args &&... args)                                                       \
    {                                                                          \
        return _name##_operation<Args...>(std::forward<Args>(args)...);        \
    }


namespace {


template <typename ValueType>
struct TemplatedGemmOperation {
    GINKGO_REGISTER_OPERATION(gemm, gemm<ValueType>);
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
GINKGO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
