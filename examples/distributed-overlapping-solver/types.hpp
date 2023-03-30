/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


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
