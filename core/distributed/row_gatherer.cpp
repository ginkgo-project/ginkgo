// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/row_gatherer.hpp"

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/vector.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/event_kernels.hpp"

namespace gko {
namespace experimental {
namespace distributed {


namespace event {
namespace {
GKO_REGISTER_OPERATION(record_event, event::record_event);
}
}  // namespace event


template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_async(
    ptr_param<const AbstractMultiVector> b,
    ptr_param<AbstractMultiVector> x) const
{
    return apply_async(b, x, send_cache_);
}


template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_async(
    ptr_param<const AbstractMultiVector> b, ptr_param<AbstractMultiVector> x,
    gko::detail::GenericDenseCache& workspace) const
{
    auto ev = this->apply_prepare(b, workspace);
    return this->apply_finalize(b, x, ev, workspace);
}

template <typename LocalIndexType>
std::shared_ptr<const gko::detail::Event>
RowGatherer<LocalIndexType>::apply_prepare(
    ptr_param<const AbstractMultiVector> b) const
{
    return apply_prepare(b, send_cache_);
}

template <typename LocalIndexType>
std::shared_ptr<const gko::detail::Event>
RowGatherer<LocalIndexType>::apply_prepare(
    ptr_param<const AbstractMultiVector> b,
    gko::detail::GenericDenseCache& workspace) const
{
    std::shared_ptr<const gko::detail::Event> ev = nullptr;
    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;
    auto tmp_b = make_temporary_clone(exec, b);

    std::visit(
        [&](auto p) {
            using ValueType = std::decay_t<decltype(p)>;

            auto b_local =
                as<Vector<ValueType>>(tmp_b.get())->get_local_vector();

            dim<2> send_size(coll_comm_->get_send_size(),
                             b_local->get_size()[1]);
            auto send_buffer = workspace.get<ValueType>(mpi_exec, send_size);
            b_local->row_gather(&send_idxs_, send_buffer);
            b_local->get_executor()->run(event::make_record_event(ev));
        },
        precision_to_variant(b->get_precision()));
    return ev;
}


template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_finalize(
    ptr_param<const AbstractMultiVector> b, ptr_param<AbstractMultiVector> x,
    std::shared_ptr<const gko::detail::Event> ev) const
{
    auto req = apply_finalize(b, x, ev, send_cache_);
    return req;
}

template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_finalize(
    ptr_param<const AbstractMultiVector> b, ptr_param<AbstractMultiVector> x,
    std::shared_ptr<const gko::detail::Event> ev,
    gko::detail::GenericDenseCache& workspace) const
{
    mpi::request req;

    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

    GKO_THROW_IF_INVALID(
        !use_host_buffer || mpi_exec->memory_accessible(x->get_executor()),
        "The receive buffer uses device memory, but MPI support of device "
        "memory is not available or host buffer were explicitly requested. "
        "Please provide a host buffer or enable MPI support for device "
        "memory.");

    auto tmp_b = make_temporary_clone(exec, b);
    std::visit(
        [&](auto p) {
            using ValueType = std::decay_t<decltype(p)>;

            auto b_local =
                as<Vector<ValueType>>(tmp_b.get())->get_local_vector();

            dim<2> send_size(coll_comm_->get_send_size(),
                             b_local->get_size()[1]);
            auto send_buffer = workspace.get<ValueType>(mpi_exec, send_size);

            auto x_global = as<Vector<ValueType>>(x->as_precision(b_local));
            auto recv_ptr = x_global->get_local_values();
            auto send_ptr = send_buffer->get_values();
            ev->synchronize();
            mpi::contiguous_type type(b_local->get_size()[1],
                                      mpi::type_impl<ValueType>::get_type());
            req = coll_comm_->i_all_to_all_v(mpi_exec, send_ptr, type.get(),
                                             recv_ptr, type.get());
        },
        precision_to_variant(b->get_precision()));

    return req;
}


namespace detail {


template <typename LocalIndexType>
std::shared_ptr<const gko::detail::Event> apply_prepare(
    const RowGatherer<LocalIndexType>* rg,
    ptr_param<const AbstractMultiVector> b)
{
    return rg->apply_prepare(b);
}


template <typename LocalIndexType>
std::shared_ptr<const gko::detail::Event> apply_prepare(
    const RowGatherer<LocalIndexType>* rg,
    ptr_param<const AbstractMultiVector> b,
    gko::detail::GenericDenseCache& workspace)
{
    return rg->apply_prepare(b, workspace);
}


template <typename LocalIndexType>
mpi::request apply_finalize(const RowGatherer<LocalIndexType>* rg,
                            ptr_param<const AbstractMultiVector> b,
                            ptr_param<AbstractMultiVector> x,
                            std::shared_ptr<const gko::detail::Event> ev)
{
    return rg->apply_finalize(b, x, ev);
}


template <typename LocalIndexType>
mpi::request apply_finalize(const RowGatherer<LocalIndexType>* rg,
                            ptr_param<const AbstractMultiVector> b,
                            ptr_param<AbstractMultiVector> x,
                            std::shared_ptr<const gko::detail::Event> ev,
                            gko::detail::GenericDenseCache& workspace)
{
    return rg->apply_finalize(b, x, ev, workspace);
}


#define GKO_DECLARE_TEST_APPLY_PREPARE(IndexType)            \
    std::shared_ptr<const gko::detail::Event> apply_prepare( \
        const RowGatherer<IndexType>*, ptr_param<const AbstractMultiVector>)

#define GKO_DECLARE_TEST_APPLY_PREPARE_WORKSPACE(IndexType)                  \
    std::shared_ptr<const gko::detail::Event> apply_prepare(                 \
        const RowGatherer<IndexType>*, ptr_param<const AbstractMultiVector>, \
        gko::detail::GenericDenseCache&)

#define GKO_DECLARE_TEST_APPLY_FINALIZE(IndexType)                      \
    mpi::request apply_finalize(const RowGatherer<IndexType>* rg,       \
                                ptr_param<const AbstractMultiVector> b, \
                                ptr_param<AbstractMultiVector> x,       \
                                std::shared_ptr<const gko::detail::Event> ev)

#define GKO_DECLARE_TEST_APPLY_FINALIZE_WORKSPACE(IndexType)                  \
    mpi::request apply_finalize(const RowGatherer<IndexType>* rg,             \
                                ptr_param<const AbstractMultiVector> b,       \
                                ptr_param<AbstractMultiVector> x,             \
                                std::shared_ptr<const gko::detail::Event> ev, \
                                gko::detail::GenericDenseCache&)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_TEST_APPLY_PREPARE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_TEST_APPLY_PREPARE_WORKSPACE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_TEST_APPLY_FINALIZE);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_TEST_APPLY_FINALIZE_WORKSPACE);

#undef GKO_DECLARE_TEST_APPLY_PREPARE
#undef GKO_DECLARE_TEST_APPLY_PREPARE_WORKSPACE
#undef GKO_DECLARE_TEST_APPLY_FINALIZE
#undef GKO_DECLARE_TEST_APPLY_FINALIZE_WORKSPACE


}  // namespace detail


template <typename LocalIndexType>
dim<2> RowGatherer<LocalIndexType>::get_size() const
{
    return size_;
}


template <typename LocalIndexType>
std::shared_ptr<const mpi::CollectiveCommunicator>
RowGatherer<LocalIndexType>::get_collective_communicator() const
{
    return coll_comm_;
}


template <typename T>
T global_add(std::shared_ptr<const Executor> exec,
             const mpi::communicator& comm, const T& value)
{
    T result;
    comm.all_reduce(std::move(exec), &value, &result, 1, MPI_SUM);
    return result;
}


template <typename LocalIndexType>
template <typename GlobalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : PolymorphicObject(exec),
      DistributedBase(coll_comm->get_base_communicator()),
      size_(dim<2>{global_add(exec, coll_comm->get_base_communicator(),
                              imap.get_non_local_size()),
                   imap.get_global_size()}),
      coll_comm_(std::move(coll_comm)),
      send_idxs_(exec),
      send_cache_()
{
    // check that the coll_comm_ and imap have the same recv size
    // the same check for the send size is not possible, since the
    // imap doesn't store send indices
    GKO_THROW_IF_INVALID(
        coll_comm_->get_recv_size() == imap.get_non_local_size(),
        "The collective communicator doesn't match the index map.");

    auto comm = coll_comm_->get_base_communicator();
    auto inverse_comm = coll_comm_->create_inverse();

    auto mpi_exec =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator())
            ? exec->get_master()
            : exec;
    auto temp_remote_local_idxs =
        make_temporary_clone(mpi_exec, &imap.get_remote_local_idxs());

    send_idxs_.set_executor(mpi_exec);
    send_idxs_.resize_and_reset(coll_comm_->get_send_size());
    inverse_comm
        ->i_all_to_all_v(exec, temp_remote_local_idxs->get_const_flat_data(),
                         send_idxs_.get_data())
        .wait();
    send_idxs_.set_executor(exec);
}


template <typename LocalIndexType>
const LocalIndexType* RowGatherer<LocalIndexType>::get_const_send_idxs() const
{
    return send_idxs_.get_const_data();
}


template <typename LocalIndexType>
size_type RowGatherer<LocalIndexType>::get_num_send_idxs() const
{
    return send_idxs_.get_size();
}


template <typename LocalIndexType>
std::unique_ptr<RowGatherer<LocalIndexType>>
RowGatherer<LocalIndexType>::create(std::shared_ptr<const Executor> exec,
                                    mpi::communicator comm)
{
    return std::unique_ptr<RowGatherer>(new RowGatherer(
        std::move(exec),
        mpi::detail::create_default_collective_communicator(comm)));
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                         mpi::communicator comm)
    : RowGatherer(std::move(exec),
                  mpi::detail::create_default_collective_communicator(comm))
{}


template <typename LocalIndexType>
std::unique_ptr<RowGatherer<LocalIndexType>>
RowGatherer<LocalIndexType>::create(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm_template)
{
    return std::unique_ptr<RowGatherer>(
        new RowGatherer(std::move(exec), std::move(coll_comm_template)));
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm_template)
    : PolymorphicObject(exec),
      DistributedBase(coll_comm_template->get_base_communicator()),
      coll_comm_(std::move(coll_comm_template)),
      send_idxs_(exec)
{}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(RowGatherer&& o) noexcept
    : PolymorphicObject(o.get_executor()),
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
        size_ = o.get_size();
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
        size_ = std::exchange(o.size_, dim<2>{});
        coll_comm_ = std::exchange(
            o.coll_comm_, mpi::detail::create_default_collective_communicator(
                              o.get_communicator()));
        send_idxs_ = std::move(o.send_idxs_);
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(const RowGatherer& o)
    : PolymorphicObject(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor())
{
    *this = o;
}


#define GKO_DECLARE_ROW_GATHERER(IndexType) class RowGatherer<IndexType>

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
