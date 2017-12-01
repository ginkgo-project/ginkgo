#include "core/matrix/csr.hpp"


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(const LinOp *other)
{
    // TODO
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    // TODO
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
    return std::unique_ptr<LinOp>(nullptr);  // TODO
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::clear()
{
    // TODO
}


#define DECLARE_CSR_MATRIX(ValueType, IndexType) class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_CSR_MATRIX);
#undef DECLARE_CSR_MATRIX


}  // namespace matrix
}  // namespace gko
