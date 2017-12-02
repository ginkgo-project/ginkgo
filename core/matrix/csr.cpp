#include "core/matrix/csr.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(const LinOp *other)
{
    auto convertible_to_csr =
        dynamic_cast<const ConvertibleTo<Csr<ValueType, IndexType>> *>(other);
    if (convertible_to_csr == nullptr) {
        throw NOT_SUPPORTED(other);
    }
    convertible_to_csr->convert_to(this);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    auto convertible_to_csr =
        dynamic_cast<ConvertibleTo<Csr<ValueType, IndexType>> *>(other.get());
    if (convertible_to_csr == nullptr) {
        throw NOT_SUPPORTED(other);
    }
    convertible_to_csr->move_to(this);
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
                this->get_num_cols(), this->get_num_nonzeros()));
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
    *other = *this;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Csr *other)
{
    *other = std::move(*this);
}


#define DECLARE_CSR_MATRIX(ValueType, IndexType) class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_CSR_MATRIX);
#undef DECLARE_CSR_MATRIX


}  // namespace matrix
}  // namespace gko
