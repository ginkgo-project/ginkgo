#include "core/matrix/csr.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(const LinOp *other)
{
    as<ConvertibleTo<Csr<ValueType, IndexType>>>(other)->convert_to(this);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    as<ConvertibleTo<Csr<ValueType, IndexType>>>(other.get())->move_to(this);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
{
    // TODO
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    // TODO
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::clone_type() const
{
    return std::unique_ptr<LinOp>(
        new Csr(this->get_executor(), this->get_num_rows(),
                this->get_num_cols(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::clear()
{
    this->set_dimensions(0, 0, 0);
    values_.clear();
    col_idxs_.clear();
    row_ptrs_.clear();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(Csr *other) const
{
    other->set_dimensions(this->get_num_rows(), this->get_num_cols(),
                          this->get_num_stored_elements());
    other->values_ = values_;
    other->col_idxs_ = col_idxs_;
    other->row_ptrs_ = row_ptrs_;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Csr *other)
{
    other->set_dimensions(this->get_num_rows(), this->get_num_cols(),
                          this->get_num_stored_elements());
    other->values_ = std::move(values_);
    other->col_idxs_ = std::move(col_idxs_);
    other->row_ptrs_ = std::move(row_ptrs_);
}


#define DECLARE_CSR_MATRIX(ValueType, IndexType) class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_CSR_MATRIX);
#undef DECLARE_CSR_MATRIX


}  // namespace matrix
}  // namespace gko
