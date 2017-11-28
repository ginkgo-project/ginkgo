#ifndef GINKGO_CORE_MATRIX_DENSE_HPP_
#define GINKGO_CORE_MATRIX_DENSE_HPP_


#include "core/base/array.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"


#include <complex>
#include <memory>


namespace gko {


template <typename ResultType>
class ConvertibleTo {
public:
    using result_type = ResultType;

    virtual void convert_to(result_type *result) const = 0;
    virtual void move_to(result_type *result) = 0;
};


namespace matrix {


template <typename ValueType>
class Dense : public LinOp, public ConvertibleTo<Dense<ValueType>> {
public:
    using value_type = ValueType;

    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         size_type num_rows, size_type num_cols,
                                         size_type padding)
    {
        return std::unique_ptr<Dense>(
            new Dense(std::move(exec), num_rows, num_cols, padding));
    }

    value_type *get_values() noexcept { return values_.get_data(); }

    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    size_type get_padding() const { return padding_; }

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(full_precision alpha, const LinOp *b, full_precision beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(Dense *result) const override;

    void move_to(Dense *result) override;

protected:
    Dense(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols, size_type padding)
        : LinOp(exec, num_rows, num_cols, num_rows * padding),
          values_(exec, num_rows * padding),
          padding_(padding)
    {}

private:
    Array<value_type> values_;
    size_type padding_;
};


}  // namespace matrix


namespace kernels {


#define GINKGO_DECLARE_GEMM_KERNEL(_type)                 \
    void gemm(_type alpha, const matrix::Dense<_type> *a, \
              const matrix::Dense<_type> *b, _type beta,  \
              matrix::Dense<_type> *c)


namespace cpu {


template <typename ValueType>
GINKGO_DECLARE_GEMM_KERNEL(ValueType);


}  // namespace cpu


namespace gpu {


template <typename ValueType>
GINKGO_DECLARE_GEMM_KERNEL(ValueType);


}  // namespace gpu


}  // namespace kernels


}  // namespace gko


#endif  // GINKGO_CORE_MATRIX_DENSE_HPP_
