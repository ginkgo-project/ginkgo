#ifndef GKO_CORE_MATRIX_DENSE_HPP_
#define GKO_CORE_MATRIX_DENSE_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"


#include <complex>
#include <initializer_list>
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

    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         size_type padding,
                                         std::initializer_list<ValueType> vals)
    {
        int num_rows = vals.size();
        std::unique_ptr<Dense> tmp(
            new Dense(exec->get_master(), num_rows, 1, padding));
        size_type idx = 0;
        for (const auto &elem : vals) {
            tmp->at(idx) = elem;
            ++idx;
        }
        auto result = create(std::move(exec));
        result->copy_from(std::move(tmp));
        return result;
    }

    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         std::initializer_list<ValueType> vals)
    {
        return create(std::move(exec), 1, vals);
    }

    static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec, size_type padding,
        std::initializer_list<std::initializer_list<ValueType>> vals)
    {
        int num_rows = vals.size();
        int num_cols = num_rows > 0 ? begin(vals)->size() : 1;
        std::unique_ptr<Dense> tmp(
            new Dense(exec->get_master(), num_rows, num_cols, padding));
        size_type ridx = 0;
        for (const auto &row : vals) {
            size_type cidx = 0;
            for (const auto &elem : row) {
                tmp->at(ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
        auto result = create(std::move(exec));
        result->copy_from(std::move(tmp));
        return result;
    }

    static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec,
        std::initializer_list<std::initializer_list<ValueType>> vals)
    {
        using std::max;
        return create(
            std::move(exec),
            vals.size() > 0 ? max<size_type>(begin(vals)->size(), 1) : 1, vals);
    }

    Array<value_type> &get_values() noexcept { return values_; }

    const Array<value_type> &get_values() const noexcept { return values_; }

    size_type get_padding() const { return padding_; }

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    virtual void scale(const LinOp *alpha);

    virtual void add_scaled(const LinOp *alpha, const LinOp *b);

    virtual void compute_dot(const LinOp *b, LinOp *result) const;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(Dense *result) const override;

    void move_to(Dense *result) override;

    ValueType &at(size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    ValueType at(size_type row, size_type col) const noexcept
    {
        return values_.get_const_data()[linearize_index(row, col)];
    }

    ValueType &at(size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(idx)];
    }

    ValueType at(size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(idx)];
    }

protected:
    Dense(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols, size_type padding)
        : LinOp(exec, num_rows, num_cols, num_rows * padding),
          values_(exec, num_rows * padding),
          padding_(padding)
    {}

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row * padding_ + col;
    }

    size_type linearize_index(size_type idx) const noexcept
    {
        return linearize_index(idx / this->get_num_cols(),
                               idx % this->get_num_cols());
    }

private:
    Array<value_type> values_;
    size_type padding_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_HPP_
