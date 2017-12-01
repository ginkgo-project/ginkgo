#ifndef GKO_CORE_MATRIX_CSR_HPP_
#define GKO_CORE_MATRIX_CSR_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"


namespace gko {
namespace matrix {


template <typename ValueType = default_precision, typename IndexType = int32>
class Csr : public LinOp {
public:
    using value_type = ValueType;

    using index_type = IndexType;

    static std::unique_ptr<Csr> create(std::shared_ptr<const Executor> exec,
                                       size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros)
    {
        return std::unique_ptr<Csr>(
            new Csr(exec, num_rows, num_cols, num_nonzeros));
    }

    static std::unique_ptr<Csr> create(std::shared_ptr<const Executor> exec)
    {
        return create(exec, 0, 0, 0);
    }

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    Array<value_type> &get_values() noexcept { return values_; }

    const Array<value_type> &get_values() const noexcept { return values_; }

    Array<index_type> &get_col_idxs() noexcept { return col_idxs_; }

    const Array<index_type> &get_col_idxs() const noexcept { return col_idxs_; }

    Array<index_type> &get_row_ptrs() noexcept { return row_ptrs_; }

    const Array<index_type> &get_row_ptrs() const noexcept { return row_ptrs_; }

protected:
    Csr(std::shared_ptr<const Executor> exec, size_type num_rows,
        size_type num_cols, size_type num_nonzeros)
        : LinOp(exec, num_rows, num_cols, num_nonzeros),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          row_ptrs_(exec, num_rows + (num_rows > 0))  // avoid allocation for 0
    {}

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<index_type> row_ptrs_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_HPP_
