// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/row_gatherer.hpp"

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"

namespace gko {
namespace experimental {
namespace distributed {


#if GINKGO_HAVE_OPENMPI_POST_4_1_X
using DefaultCollComm = mpi::NeighborhoodCommunicator;
#else
using DefaultCollComm = mpi::DenseCommunicator;
#endif


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
std::shared_ptr<const mpi::CollectiveCommunicator>
RowGatherer<LocalIndexType>::get_collective_communicator() const
{
    return coll_comm_;
}


template <typename LocalIndexType>
template <typename GlobalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : EnableLinOp<RowGatherer>(
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
        ->i_all_to_all_v(exec,
                         imap.get_remote_local_idxs().get_const_flat_data(),
                         send_idxs_.get_data())
        .wait();
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                         mpi::communicator comm)
    : EnableLinOp<RowGatherer>(exec),
      DistributedBase(comm),
      coll_comm_(std::make_shared<DefaultCollComm>(comm)),
      send_idxs_(exec)
{}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(RowGatherer&& o) noexcept
    : EnableLinOp<RowGatherer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor())
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
            o.coll_comm_,
            std::make_shared<DefaultCollComm>(o.get_communicator()));
        send_idxs_ = std::move(o.send_idxs_);
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(const RowGatherer& o)
    : EnableLinOp<RowGatherer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor())
{
    *this = o;
}


#define GKO_DECLARE_ROW_GATHERER(_itype) class RowGatherer<_itype>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_GATHERER);

#undef GKO_DECLARE_ROW_GATHERER


#define GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR(_ltype, _gtype)          \
    RowGatherer<_ltype>::RowGatherer(                                 \
        std::shared_ptr<const Executor> exec,                         \
        std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm, \
        const index_map<_ltype, _gtype>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR);

#undef GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
