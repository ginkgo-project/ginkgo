

#ifndef GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_TYPES_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_TYPES_HPP


#include <ginkgo/ginkgo.hpp>


// @sect3{Type Definitiions}
// Define the needed types. In a parallel program we need to differentiate
// beweeen global and local indices, thus we have two index types.
using LocalIndexType = gko::int32;
// The underlying value type.
using ValueType = double;
// As vector type we use the following, which implements a subset of @ref
// gko::matrix::Dense.
using vec = gko::matrix::Dense<ValueType>;
using dist_vec = gko::experimental::distributed::Vector<ValueType>;
// As matrix type we simply use the following type, which can read
// distributed data and be applied to a distributed vector.
using mtx = gko::matrix::Csr<ValueType, LocalIndexType>;
using dist_mtx =
    gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                           LocalIndexType>;
// We can use here the same solver type as you would use in a
// non-distributed program. Please note that not all solvers support
// distributed systems at the moment.
using solver = gko::solver::Cg<ValueType>;


struct shared_idx_t {
    int local_idx;
    // the index within the local indices of the remote_rank
    int remote_idx;
    // rank that shares the local DOF
    int remote_rank;
    // can be used in non-overlapping case for interface DOFs to denote keep
    // these DOFs in local view
    int owning_rank;
};


#endif  // GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_TYPES_HPP
