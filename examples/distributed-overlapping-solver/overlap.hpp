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

#ifndef GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP

#include <ginkgo/ginkgo.hpp>
#include "core/base/iterator_factory.hpp"

#include "types.hpp"


template <typename Vec>
std::pair<int, double> cg(gko::ptr_param<const gko::LinOp> op,
                          gko::ptr_param<const gko::LinOp> pre,
                          gko::ptr_param<const Vec> b, gko::ptr_param<Vec> x,
                          int max_it, double reduction)
{
    auto exec = op->get_executor();
    auto r = b->clone();
    auto z = b->clone();

    auto rho = gko::initialize<vec>({0.0}, exec);
    auto alpha = gko::initialize<vec>({0.0}, exec);
    auto pAq = gko::initialize<vec>({0.0}, exec);
    auto beta = gko::initialize<vec>({0.0}, exec);
    auto rho_old = rho->clone();

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    op->apply(neg_one, x, one, r);

    // need to make sure that z has zeros on dirichlet dofs
    z->fill(0.0);
    pre->apply(r, z);

    auto p = z->clone();
    auto q = b->clone();

    r->compute_dot(z, rho);

    for (int i = 0; i < max_it; ++i) {
        op->apply(p, q);

        p->compute_dot(q, pAq);

        alpha->copy_from(rho);
        alpha->inv_scale(pAq);

        x->add_scaled(alpha, p);
        r->sub_scaled(alpha, q);

        z->fill(0.0);
        pre->apply(r, z);

        std::swap(rho, rho_old);
        r->compute_dot(z, rho);

        if (rho->at(0) < reduction) {
            return {i, rho->at(0)};
        }

        beta->copy_from(rho);
        beta->inv_scale(rho_old);

        z->add_scaled(beta, p);
        std::swap(p, z);
    }

    return {max_it, rho->at(0)};
}

struct set {
    template <typename ValueType>
    ValueType operator()(const ValueType& local, const ValueType& remote)
    {
        return remote;
    }
};
struct add {
    template <typename ValueType>
    ValueType operator()(const ValueType& local, const ValueType& remote)
    {
        return local + remote;
    }
};


namespace gko {


struct neighborhood_descriptor {
    neighborhood_descriptor() {}

    std::vector<int> send_sizes;
    std::vector<int> send_offsets;
    std::vector<int> recv_sizes;
    std::vector<int> recv_offsets;
};


/**
 * Struct to hold all necessary information for an all-to-all communication
 * on vectors with shared DOFs.
 *
 * Specialize this to partition DOFs according to if they are shared or not.
 * This would store first all locally owned DOFs, and then all non-owned/shared.
 * It would then be possible to skip the row_gather/inv_row_gather and use the
 * partitioned memory as send/recv buffers directly.
 * Note: this will create a reordering of the DOFs.
 */
struct comm_info_t {
    comm_info_t() = default;

    /**
     * Extracts communication pattern from a list of shared DOFs.
     *
     * @param comm
     * @param shared_idxs
     */
    comm_info_t(experimental::mpi::communicator comm,
                const array<shared_idx_t>& shared_idxs)
        : send_sizes(comm.size()),
          send_offsets(comm.size() + 1),
          recv_sizes(comm.size()),
          recv_offsets(comm.size() + 1),
          recv_idxs(shared_idxs.get_executor(), shared_idxs.get_num_elems()),
          send_idxs(shared_idxs.get_executor()),
          multiplicity{gko::array<LocalIndexType>{shared_idxs.get_executor()},
                       gko::array<ValueType>{shared_idxs.get_executor()}}
    {
        auto exec = shared_idxs.get_executor()->get_master();
        std::vector<int> remote_idxs(shared_idxs.get_num_elems());
        std::vector<int> recv_ranks(shared_idxs.get_num_elems());

        std::map<LocalIndexType, ValueType> weight_map;

        // this is basically AOS->SOA but with just counting the remote ranks
        for (int i = 0; i < shared_idxs.get_num_elems(); ++i) {
            auto& shared_idx = shared_idxs.get_const_data()[i];
            recv_sizes[shared_idx.remote_rank]++;

            recv_ranks[i] = shared_idx.remote_rank;
            recv_idxs.get_data()[i] = shared_idx.local_idx;

            remote_idxs[i] = shared_idx.remote_idx;

            if (shared_idx.owning_rank != comm.rank()) {
                auto it = weight_map.find(shared_idx.local_idx);
                if (it == weight_map.end()) {
                    weight_map[shared_idx.local_idx] = 0.0;
                }
            } else {
                auto it = weight_map.find(shared_idx.local_idx);
                if (it == weight_map.end()) {
                    weight_map[shared_idx.local_idx] = 1.0;
                }
                weight_map[shared_idx.local_idx] += 1.0;
            }
        }
        multiplicity.idxs.resize_and_reset(weight_map.size());
        multiplicity.weights.resize_and_reset(weight_map.size());
        {
            int i = 0;
            for (const auto& elem : weight_map) {
                const auto& idx = elem.first;
                const auto& weight = elem.second;
                multiplicity.idxs.get_data()[i] = idx;
                // need sqrt for norms and scalar products
                if (weight == 0.0) {
                    multiplicity.weights.get_data()[i] = 1.0;
                } else {
                    multiplicity.weights.get_data()[i] = sqrt(1.0 / weight);
                }
                ++i;
            }
        }
        // sort by rank
        auto sort_it = detail::make_zip_iterator(
            recv_ranks.data(), recv_idxs.get_data(), remote_idxs.data());
        std::sort(sort_it, sort_it + shared_idxs.get_num_elems(),
                  [](const auto a, const auto b) {
                      return std::get<0>(a) < std::get<0>(b);
                  });

        std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                         recv_offsets.begin() + 1);

        // exchange recv_idxs to get which indices this rank has to sent to
        // every other rank
        comm.all_to_all(exec, recv_sizes.data(), 1, send_sizes.data(), 1);
        std::partial_sum(send_sizes.begin(), send_sizes.end(),
                         send_offsets.begin() + 1);
        send_idxs.resize_and_reset(send_offsets.back());

        comm.all_to_all_v(exec, remote_idxs.data(), recv_sizes.data(),
                          recv_offsets.data(), send_idxs.get_data(),
                          send_sizes.data(), send_offsets.data());
    }

    ValueType get_weight(LocalIndexType idx) const
    {
        auto it = std::lower_bound(multiplicity.idxs.get_const_data(),
                                   multiplicity.idxs.get_const_data() +
                                       multiplicity.idxs.get_num_elems(),
                                   idx);
        auto dist = std::distance(multiplicity.idxs.get_const_data(), it);
        if (dist < multiplicity.idxs.get_num_elems() && *it == idx) {
            return multiplicity.weights.get_const_data()[dist];
        } else {
            return one<ValueType>();
        }
    }

    // default variable all-to-all data
    std::vector<int> send_sizes;
    std::vector<int> send_offsets;
    std::vector<int> recv_sizes;
    std::vector<int> recv_offsets;

    // DOFs to send. These are dofs that are shared with other ranks,
    // but this rank owns them.
    gko::array<LocalIndexType> send_idxs;
    // DOFs to send. These are dofs that are shared with other ranks,
    // but other ranks own them. May overlap with send_idxs.
    gko::array<LocalIndexType> recv_idxs;
    // maybe also store multiplicity of each index?
    // could also store explicit zero for non-owned idxs
    // non-owned idxs have explicit zero weight.
    struct multiplicity_t {
        gko::array<LocalIndexType> idxs;
        gko::array<ValueType> weights;
    };
    multiplicity_t multiplicity;
};


/**
 * Currently this class is explicitly for overlapping methods. But it should be
 * possible to use it also for non-overlapping methods. It is only defined by
 * the fact that there is no global partition or other information. Everything
 * is handled purely locally, also wrt to commmunication (ie only local indices
 * are used in the communication).
 */
struct overlapping_vec : public EnableLinOp<overlapping_vec, vec> {
    overlapping_vec(std::shared_ptr<const Executor> exec)
        : EnableLinOp<overlapping_vec, vec>(exec), comm(MPI_COMM_NULL)
    {}

    overlapping_vec(std::shared_ptr<const Executor> exec,
                    experimental::mpi::communicator comm,
                    std::unique_ptr<vec> local_vec, comm_info_t comm_info)
        : EnableLinOp<overlapping_vec, vec>(local_vec->get_executor(),
                                            local_vec->get_size()),
          comm(comm),
          comm_info(comm_info)
    {
        static_cast<vec&>(*this) = std::move(*local_vec);
    }

    std::unique_ptr<Dense> create_with_same_config() const override
    {
        return std::make_unique<overlapping_vec>(
            this->get_executor(), this->comm, vec::create_with_same_config(),
            comm_info);
    }


    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    enum class operation { copy, add, average };

    /**
     * Updates non-local dofs from their respective ranks, using
     * the update scheme (copy, add) specified for each dof.
     *
     * Considering a distributed operator A = sum_i R_i^T D_i A_i R_i, this
     * will peform u_i = R_i R_i^T D_i u_i, eliminating the need for a
     * fully global vector
     *
     * The update operation corresponds to the entries of D_i for the shared
     * DOFs. Currently, copy results in D_i = 0 for non_owned_idxs and D_i =
     * 1/#shared_ranks for owned, but shared DOFS. The D_i = 1/#shared_ranks is
     * implicit, because it is assumed that the vector contains the same value
     * on these indices for all sharing ranks. This is the case in overlapping
     * FEM applications, because the local stiffness matrices will have the same
     * values for rows with DOFs that are shared between ranks and in the
     * interior of all overlapping domains.
     * Other D_i, for example D_i = 1/#shared_ranks for all shared DOFs (even
     * locally owned), would require more information than currently available.
     * This would require specifying DOFs that are locally owned, but shared
     * with other ranks explicitly as shared_dofs during the creation of
     * comm_info, as well as combining the sets recv_idxs and send_idxs.
     *
     * The other operations are not implemented yet.
     *
     * Note that all point-wise vector operations (adding, scaling, etc) keep
     * the consistency of the vector. Only operations where the update for an
     * entry i depends on other entries than i can make a vector non-consistent.
     * This usually applies for all operator applications.
     *
     * Perhaps something more generic to allow for user-defined transformations?
     */
    void make_consistent(operation op)
    {
        auto exec = this->get_executor();
        auto send_buffer =
            vec::create(exec, dim<2>(comm_info.send_offsets.back(), 1));
        auto recv_buffer =
            vec::create(exec, dim<2>(comm_info.recv_offsets.back(), 1));

        if (op == operation::copy) {
            if (comm_info.send_offsets.back() > 0) {
                // can't handle row gather with empty idxs??
                this->row_gather(&comm_info.send_idxs, send_buffer.get());
                comm.all_to_all_v(
                    exec, send_buffer->get_values(),
                    comm_info.send_sizes.data(), comm_info.send_offsets.data(),
                    recv_buffer->get_values(), comm_info.recv_sizes.data(),
                    comm_info.recv_offsets.data());
                // inverse row_gather
                // unnecessary if shared_idxs would be stored separately
                for (int i = 0; i < comm_info.recv_idxs.get_num_elems(); ++i) {
                    this->at(comm_info.recv_idxs.get_data()[i]) =
                        recv_buffer->at(i);
                }
            }
        } else if (op == operation::average) {
            if (comm_info.send_offsets.back() > 0) {
                // can't handle row gather with empty idxs??
                this->row_gather(&comm_info.send_idxs, send_buffer.get());
                comm.all_to_all_v(
                    exec, send_buffer->get_values(),
                    comm_info.send_sizes.data(), comm_info.send_offsets.data(),
                    recv_buffer->get_values(), comm_info.recv_sizes.data(),
                    comm_info.recv_offsets.data());
                // inverse row_gather
                // unnecessary if shared_idxs would be stored separately
                for (int i = 0; i < comm_info.recv_idxs.get_num_elems(); ++i) {
                    this->at(comm_info.recv_idxs.get_data()[i]) =
                        (this->at(comm_info.recv_idxs.get_data()[i]) +
                         recv_buffer->at(i)) /
                        2;
                }
            }
        } else if (op == operation::add) {
            if (comm_info.send_offsets.back() > 0) {
                // can't handle row gather with empty idxs??
                this->row_gather(&comm_info.send_idxs, send_buffer.get());
                comm.all_to_all_v(
                    exec, send_buffer->get_values(),
                    comm_info.send_sizes.data(), comm_info.send_offsets.data(),
                    recv_buffer->get_values(), comm_info.recv_sizes.data(),
                    comm_info.recv_offsets.data());
                // inverse row_gather
                // unnecessary if shared_idxs would be stored separately
                for (int i = 0; i < comm_info.recv_idxs.get_num_elems(); ++i) {
                    this->at(comm_info.recv_idxs.get_data()[i]) +=
                        recv_buffer->at(i);
                }
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    /**
     * Constraints the overlapping vector to the DOFs owned by this rank. This
     * can still contain DOFs shared with other ranks, but not exclusively owned
     * by them.
     */
    std::unique_ptr<vec> extract_local() const
    {
        auto exec = this->get_executor();
        auto no_ovlp_local = vec::create(exec, dim<2>{this->get_size()[0], 1});

        // copy-if, but in stupid
        // could be row_gather(complement(idx_set))
        for (int j = 0; j < this->get_size()[0]; ++j) {
            no_ovlp_local->at(j) = this->at(j) * comm_info.get_weight(j);
        }

        return no_ovlp_local;
    }

    void compute_dot_impl(const LinOp* b, LinOp* result) const override
    {
        auto ovlp_b = dynamic_cast<const overlapping_vec*>(b);
        auto no_ovlp_b = ovlp_b->extract_local();
        auto no_ovlp_local = this->extract_local();

        auto dist_b =
            dist_vec::create(no_ovlp_b->get_executor(), comm, no_ovlp_b.get());
        auto dist_local = dist_vec::create(no_ovlp_local->get_executor(), comm,
                                           no_ovlp_local.get());

        dist_local->compute_dot(dist_b.get(), result);
    }

    void compute_norm2_impl(LinOp* result) const override
    {
        auto no_ovlp_local = extract_local();

        dist_vec::create(no_ovlp_local->get_executor(), comm,
                         no_ovlp_local.get())
            ->compute_norm2(result);
    }

    experimental::mpi::communicator comm;
    comm_info_t comm_info;
};


struct overlapping_operator
    : public experimental::distributed::DistributedBase,
      public experimental::EnableDistributedLinOp<overlapping_operator> {
    overlapping_operator(std::shared_ptr<const Executor> exec,
                         experimental::mpi::communicator comm,
                         std::shared_ptr<LinOp> local_op = nullptr,
                         comm_info_t comm_info = {},
                         overlapping_vec::operation shared_mode = {})
        : experimental::distributed::DistributedBase(comm),
          experimental::EnableDistributedLinOp<overlapping_operator>{
              exec, local_op->get_size()},
          local_op(local_op),
          comm_info(comm_info),
          shared_mode(shared_mode)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        local_op->apply(b, x);
        // exchange data
        as<overlapping_vec>(x)->make_consistent(shared_mode);
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        auto copy_x = x->clone();
        apply_impl(b, x);
        as<vec>(x)->scale(alpha);
        as<vec>(x)->add_scaled(beta, copy_x);
    }

    bool apply_uses_initial_guess() const override { return true; }

    std::shared_ptr<LinOp> local_op;
    comm_info_t comm_info;
    overlapping_vec::operation shared_mode;
};


}  // namespace gko


#endif  // GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP
