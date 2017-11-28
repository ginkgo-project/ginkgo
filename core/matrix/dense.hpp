#ifndef GKO_CORE_MATRIX_DENSE_HPP_
#define GKO_CORE_MATRIX_DENSE_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"


#include <complex>
#include <memory>


namespace gko {
namespace matrix {


template <typename ValueType = default_precision>
class Dense : public LinOp, public ConvertibleTo<Dense<ValueType>> {
public:
    using value_type = ValueType;

    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec)
    {
        return create(exec, 0, 0, 0);
    }

    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         size_type num_rows, size_type num_cols,
                                         size_type padding)
    {
        return std::unique_ptr<Dense>(
            new Dense(std::move(exec), num_rows, num_cols, padding));
    }

    Array<value_type> &get_values() noexcept { return values_; }

    const Array<value_type> &get_values() const noexcept { return values_; }

    size_type get_padding() const { return padding_; }

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(full_precision alpha, const LinOp *b, full_precision beta,
               LinOp *x) const override;

    virtual void scale(full_precision alpha);

    virtual void add_scaled(full_precision alpha, const LinOp *b);

    virtual void compute_dot(const LinOp *b, Array<full_precision> &result);

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
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_HPP_
