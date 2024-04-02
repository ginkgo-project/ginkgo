// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/row_gatherer.hpp>


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"

namespace gko {
namespace experimental {
namespace distributed {


/**
 * \brief
 * \tparam LocalIndexType index type
 * \param comm neighborhood communicator
 * \param remote_local_idxs the remote indices in their local indexing
 * \param recv_sizes the sizes that segregate remote_local_idxs
 * \param send_sizes the number of local indices per rank that are part of
 *                   remote_local_idxs on that ranks
 * \return the local indices that are part of remote_local_idxs on other ranks,
 *         ordered by the rank ordering of the communicator
 */
template <typename LocalIndexType>
array<LocalIndexType> communicate_send_gather_idxs(
    mpi::communicator comm, const array<LocalIndexType>& remote_local_idxs,
    const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes,
    const array<comm_index_type>& send_ids,
    const std::vector<comm_index_type>& send_sizes)
{
    // create temporary inverse sparse communicator
    MPI_Comm sparse_comm;
    MPI_Info info;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_create(&info));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create_adjacent(
        comm.get(), send_ids.get_size(), send_ids.get_const_data(),
        MPI_UNWEIGHTED, recv_ids.get_size(), recv_ids.get_const_data(),
        MPI_UNWEIGHTED, info, false, &sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    std::vector<comm_index_type> recv_offsets(recv_sizes.size() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes.size() + 1);
    std::partial_sum(recv_sizes.data(), recv_sizes.data() + recv_sizes.size(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes.data(), send_sizes.data() + send_sizes.size(),
                     send_offsets.begin() + 1);

    array<LocalIndexType> send_gather_idxs(remote_local_idxs.get_executor(),
                                           send_offsets.back());

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Neighbor_alltoallv(
        remote_local_idxs.get_const_data(), recv_sizes.data(),
        recv_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        send_gather_idxs.get_data(), send_sizes.data(), send_offsets.data(),
        mpi::type_impl<LocalIndexType>::get_type(), sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(&sparse_comm));

    return send_gather_idxs;
}


template <typename LocalIndexType>
void RowGatherer<LocalIndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    apply_async(b, x).wait();
}


template <typename LocalIndexType>
void RowGatherer<LocalIndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
    GKO_NOT_IMPLEMENTED;


template <typename LocalIndexType>
std::future<void> RowGatherer<LocalIndexType>::apply_async(
    ptr_param<const LinOp> b, ptr_param<LinOp> x) const
{
    auto op = [b = b.get(), x = x.get(), rg = this->shared_from_this(),
               id = current_id_++] {
        // ensure that the communications are executed in the order
        // the apply_async were called
        while (id > rg->active_id_.load()) {
            std::this_thread::yield();
        }

        // dispatch global vector
        run<Vector, double, float, std::complex<double>, std::complex<float>>(
            b, [&](const auto* b_global) {
                using ValueType =
                    typename std::decay_t<decltype(*b_global)>::value_type;
                // dispatch local vector with the same precision as the global
                // vector
                ::gko::precision_dispatch<ValueType>(
                    [&](auto* x_local) {
                        auto exec = rg->get_executor();

                        auto b_local = b_global->get_local_vector();
                        rg->send_buffer.template init<ValueType>(
                            b_local->get_executor(),
                            dim<2>(rg->coll_comm_->get_send_size(),
                                   b_local->get_size()[1]));
                        rg->send_buffer.template get<ValueType>()->fill(0.0);
                        b_local->row_gather(
                            &rg->send_idxs_,
                            rg->send_buffer.template get<ValueType>());

                        auto recv_ptr = x_local->get_values();
                        auto send_ptr =
                            rg->send_buffer.template get<ValueType>()
                                ->get_values();

                        exec->synchronize();
                        mpi::contiguous_type type(
                            b_local->get_size()[1],
                            mpi::type_impl<ValueType>::get_type());
                        auto g = exec->get_scoped_device_id_guard();
                        auto req = rg->coll_comm_->i_all_to_all_v(
                            exec, send_ptr, type.get(), recv_ptr, type.get());
                        req.wait();
                    },
                    x);
            });

        rg->active_id_++;
    };
    return std::async(std::launch::async, op);
}

template <typename LocalIndexType>
template <typename GlobalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::collective_communicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(
          exec, dim<2>{imap.get_non_local_size(), imap.get_global_size()}),
      DistributedBase(coll_comm->get_base_communicator()),
      coll_comm_(std::move(coll_comm)),
      send_idxs_(exec)
{
    // check that the coll_comm_ and imap have the same recv size
    // the same check for the send size is not possible, since the
    // imap doesn't store send indices
    GKO_THROW_IF_INVALID(
        coll_comm_->get_recv_size() == imap.get_non_local_size(),
        "The collective communicator doesn't match the index map.");

    auto comm = coll_comm_->get_base_communicator();
    auto inverse_comm = coll_comm_->create_inverse();

    send_idxs_.resize_and_reset(coll_comm_->get_send_size());
    inverse_comm
        ->i_all_to_all_v(
            exec, imap.get_remote_local_idxs().get_flat().get_const_data(),
            send_idxs_.get_data())
        .wait();
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                         mpi::communicator comm)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(exec),
      DistributedBase(comm),
      coll_comm_(std::make_shared<mpi::neighborhood_communicator>(comm)),
      send_idxs_(exec)
{}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(RowGatherer&& o) noexcept
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(o.get_executor()),
      DistributedBase(o.get_communicator())
{
    *this = std::move(o);
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>& RowGatherer<LocalIndexType>::operator=(
    const RowGatherer& o)
{
    if (this != &o) {
        this->set_size(o.get_size());
        coll_comm_ = o.coll_comm_;
        send_idxs_ = o.send_idxs_;
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>& RowGatherer<LocalIndexType>::operator=(
    RowGatherer&& o)
{
    if (this != &o) {
        this->set_size(o.get_size());
        o.set_size({});
        coll_comm_ = std::exchange(
            o.coll_comm_, std::make_shared<mpi::neighborhood_communicator>(
                              o.get_communicator()));
        send_idxs_ = std::move(o.send_idxs_);
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(const RowGatherer& o)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(o.get_executor()),
      DistributedBase(o.get_communicator())
{
    *this = o;
}


#define GKO_DECLARE_ROW_GATHERER(_itype) class RowGatherer<_itype>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_GATHERER);

#undef GKO_DECLARE_ROW_GATHERER


#define GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR(_ltype, _gtype)           \
    RowGatherer<_ltype>::RowGatherer(                                  \
        std::shared_ptr<const Executor> exec,                          \
        std::shared_ptr<const mpi::collective_communicator> coll_comm, \
        const index_map<_ltype, _gtype>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR);

#undef GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
