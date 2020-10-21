#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {

template <typename ValueType, typename IndexType>
class DistributedMatrix {
public:
    void read(const matrix_data<ValueType, IndexType> data &)
    {
        matrix_data<ValueType, IndexType> diag_data;
        matrix_data<ValueType, IndexType> offdiag_data;
        auto cols;  // index_set for all non-local columns that occur in the
                    // off-diag matrix
        // first we need to build cols by collecting all non-zero columns from
        // rows in `partition`.
        for (auto el : data) {
            if (partition.contains(el.row)) {
                if (partition.contains(el.column)) {
                    diag_data.nonzeros.emplace_back(partition.rank(el.row),
                                                    partition.rank(el.column),
                                                    el.value);
                } else {
                    offdiag_data.nonzeros.emplace_back(
                        partition.rank(el.row), cols.rank(el.column), el.value);
                }
            }
        }
        diag_mtx_->read(diag_data);
        offdiag_mtx_->read(offdiag_data);
    }

    void apply(LinOp *b, LinOp *x) const
    {
        using Dense = matrix::Dense<ValueType>;
        // assert executor distributed
        auto dense_b = as<Dense>(b);
        auto dense_x = as<Dense>(x);
        auto local_b = dense_b->create_local_view();
        auto local_x = dense_x->create_local_view();
        // assert matching local dimensions/partition size
        diag_mtx_->apply(local_b.get(), local_x.get());
        // communicate recv_buffer_
        auto one = initialize<Dense>(exec_, {1.0});
        offdiag_mtx_->apply(one.get(), recv_buffer_.get(), one.get(), local_x);
    }

    void apply(LinOp *alpha, LinOp *b, LinOp *beta, LinOp *x) const
    {
        using Dense = matrix::Dense<ValueType>;
        // assert executor distributed
        auto dense_b = as<Dense>(b);
        auto dense_x = as<Dense>(x);
        auto local_b = dense_b->create_local_view();
        auto local_x = dense_x->create_local_view();
        // assert matching local dimensions/partition size
        diag_mtx_->apply(alpha, local_b.get(), beta, local_x.get());
        // communicate recv_buffer_
        auto one = initialize<Dense>(exec_, {1.0});
        offdiag_mtx_->apply(one.get(), recv_buffer_.get(), beta, local_x);
    }

private:
    // some gathering info
    std::shared_ptr<Executor> exec_;
    IndexSet<IndexType> partition;
    std::unique_ptr<matrix::Dense<ValueType>> recv_buffer_;
    std::unique_ptr<LinOp> diag_mtx_;
    std::unique_ptr<LinOp> offdiag_mtx_;
}

}  // namespace gko
