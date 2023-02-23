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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_REPARTITION_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_REPARTITION_HPP_


#include <ginkgo/config.hpp>


#ifdef GINKGO_BUILD_MPI


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>


namespace gko {
namespace experimental {
/**
 * @brief The distributed namespace.
 *
 * @ingroup distributed
 */
namespace distributed {


template <typename ValueType>
class Vector;
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
class Matrix;


template <typename LocalIndexType = int32, typename GlobalIndexType = int64>
class repartitioner : public EnableCreateMethod<
                          repartitioner<LocalIndexType, GlobalIndexType>> {
    friend class EnableCreateMethod<repartitioner>;

public:
    template <typename ValueType>
    std::tuple<gko::array<int>, gko::array<int>, gko::array<LocalIndexType>,
               std::vector<int>, std::vector<int>, std::vector<int>,
               std::vector<int>>
    gather(const Matrix<ValueType, LocalIndexType, GlobalIndexType>* from,
           Matrix<ValueType, LocalIndexType, GlobalIndexType>* to) const;

    /* updates an existing matrix without communicating the sparsity pattern **
     *
     * @param from - matrix from which coefficients should be repartitioned
     * @param to - matrix which coefficients should be overwritten
     */
    template <typename ValueType>
    void update_existing(const array<LocalIndexType>& local_indices,
                         const array<LocalIndexType>& non_local_indices,
                         const array<ValueType>& local_from_data,
                         const array<ValueType>& non_local_from_data,
                         const array<int>& sorting_idx,
                         const std::vector<int>& send_sizes,
                         const std::vector<int>& send_offs,
                         const std::vector<int>& recv_sizes,
                         const std::vector<int>& recv_offs,
                         const array<LocalIndexType>& local_scatter_pattern,
                         const array<LocalIndexType>& non_local_scatter_pattern,
                         array<ValueType>& local_to_data,
                         array<ValueType>& non_local_to_data) const;

    template <typename ValueType>
    void gather(const Vector<ValueType>* from, Vector<ValueType>* to) const;

    template <typename ValueType>
    void scatter(const Vector<ValueType>* to, Vector<ValueType>* from) const;

    mpi::communicator get_to_communicator() const { return to_comm_; }

    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
    get_from_partition() const
    {
        return from_partition_;
    }

    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
    get_to_partition() const
    {
        return to_partition_;
    }

    // check if the current rank in receives data
    bool to_has_data() const { return to_has_data_; }

protected:
    repartitioner(
        const mpi::communicator from_comm,
        std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
            from_partition,
        std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
            to_partition);

private:
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
        from_partition_;
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>>
        to_partition_;

    const mpi::communicator from_comm_;
    mpi::communicator to_comm_;

    std::shared_ptr<std::vector<comm_index_type>> default_send_sizes_;
    std::shared_ptr<std::vector<comm_index_type>> default_send_offsets_;

    std::shared_ptr<std::vector<comm_index_type>> default_recv_sizes_;
    std::shared_ptr<std::vector<comm_index_type>> default_recv_offsets_;

    bool to_has_data_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif

#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_REPARTITION_HPP_
