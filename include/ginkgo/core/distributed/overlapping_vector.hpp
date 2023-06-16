#ifndef OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP
#define OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP

#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

enum class transformation { set, add };

template <typename IndexType>
struct overlapping_partition {
    using index_type = IndexType;
    using mask_type = uint8;

    struct overlap_indices {
        using blocked = index_set<index_type>;
        using interleaved = std::vector<index_set<index_type>>;

        array<comm_index_type> target_ids;
        std::variant<blocked, interleaved> idxs;
        // store local multiplicity, i.e. if the index is owned by this process,
        // by how many is it owned in total, otherwise the multiplicity is zero
    };


    const index_set<index_type>& get_local_indices() const
    {
        return local_idxs_;
    }

    array<mask_type> get_local_mask();

    const overlap_indices& get_send_indices() const
    {
        return overlap_send_idxs_;
    }

    const overlap_indices& get_recv_indices() const
    {
        return overlap_recv_idxs_;
    }

    size_type get_overlap_num_elems(const overlap_indices& idxs) const
    {
        return std::visit(
            overloaded(
                [](const typename overlap_indices::blocked& block) {
                    return block.idxs.get_num_elems();
                },
                [](const typename overlap_indices::interleaved& interleaved) {
                    return std::accumulate(interleaved.idxs.begin(),
                                           interleaved.idxs.end(), size_type{},
                                           [](const auto& a, const auto& b) {
                                               return a + b.get_num_elems();
                                           });
                }),
            idxs.idxs);
    }

    size_type get_overlap_size(const overlap_indices& idxs) const
    {
        return std::visit(
            overloaded(
                [](const typename overlap_indices::blocked& block) {
                    return block.idxs.get_size();
                },
                [](const typename overlap_indices::interleaved& interleaved) {
                    return std::max_element(
                               interleaved.idxs.begin(), interleaved.idxs.end(),
                               [](const auto& a, const auto& b) {
                                   return a.get_size() < b.get_size();
                               })
                        ->get_size();
                }),
            idxs.idxs);
    }

    array<int8> get_overlapping_mask();

    array<index_type> get_multiplicity();

    size_type get_size()
    {
        return static_cast<size_type>(std::max(local_idxs_.get_size(),
                                               overlap_send_idxs_.get_size(),
                                               overlap_recv_idxs_.get_size()));
    }

    bool has_grouped_indices() { return group_sizes_.get_num_elems(); }

    //    index_set<index_type> get_non_local_indices();

    array<index_type> get_target_ids();

    // returns process id and overlap size
    index_type get_group_target_id(index_type group);
    index_type get_group_size(index_type group);

    /*
     * Indices are grouped first by local indices and then receiving overlapping
     * indices
     */
    static std::shared_ptr<overlapping_partition> build_from_grouped_recv(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::vector<std::pair<index_set<index_type>, comm_index_type>>
            send_idxs,
        array<index_type> target_id, array<size_type> group_size)
    {
        std::vector<index_set<index_type>> send_index_sets(send_idxs.size(),
                                                           exec);
        array<comm_index_type> send_target_ids(exec->get_master(),
                                               send_idxs.size());

        for (int i = 0; i < send_idxs.size(); ++i) {
            send_index_sets[i] = std::move(send_idxs[i].first);
            send_target_ids.get_data()[i] = send_idxs.second;
        }
        send_target_ids.set_executor(exec);

        return build_from_grouped_recv(
            std::move(exec), local_size,
            std::make_pair(std::move(send_index_sets),
                           std::move(send_target_ids)),
            std::move(target_id), std::move(group_size));
    }

    static std::shared_ptr<overlapping_partition> build_from_grouped_recv(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::pair<std::vector<index_set<index_type>>, array<comm_index_type>>
            send_idxs,
        array<index_type> target_id, array<size_type> group_size)
    {
        GKO_ASSERT(local_size >=
                   std::max_element(send_idxs.first.begin(),
                                    send_idxs.first.end(),
                                    [](const auto& a, const auto& b) {
                                        return a.get_size() < b.get_size();
                                    })
                       ->get_size());
        index_set<index_type> local_idxs(exec, {0, local_size}, true);

        auto recv_size = reduce_add(group_size);
        // need to create a subset for each target id
        index_set<index_type> recv_idxs(exec, {local_size, recv_size}, true);

        return {std::move(local_idxs),
                {std::move(send_idxs.second), std::move(send_idxs.first)},
                {std::move(target_id), std::move(recv_idxs)}};
    }

    static std::shared_ptr<overlapping_partition> build_from_arbitrary();

private:
    overlapping_partition(index_set<index_type> local_idxs,
                          overlap_indices overlap_send_idxs,
                          overlap_indices overlap_recv_idxs)
        : local_idxs_(std::move(local_idxs)),
          overlap_send_idxs_(std::move(overlap_send_idxs)),
          overlap_recv_idxs_(std::move(overlap_recv_idxs))
    {}

    // owned by this process (exclusively or shared)
    index_set<index_type> local_idxs_;
    // shared ownership by this process (subset of local_idxs_)
    overlap_indices overlap_send_idxs_;
    // not owned by this process
    overlap_indices overlap_recv_idxs_;
};

/**
 * maybe allow for processes owning multiple parts by mapping target_ids to
 * rank?
 */
template <typename IndexType>
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const overlapping_partition<IndexType>* part)
{
    auto host_exec = part->get_local_indices().get_executor()->get_master();

    auto in_degree = part->get_recv_indices().target_ids.get_num_elems();
    auto out_degree = part->get_send_indices().target_ids.get_num_elems();

    array<comm_index_type> sources{host_exec,
                                   part->get_recv_indices().target_ids};
    array<comm_index_type> destinations{host_exec,
                                        part->get_recv_indices().target_ids};

    mpi::communicator neighbor_comm{base, 0, base.rank()};
    // adjacent constructor guarantees that querying sources/destinations
    // will result in the array having the same order as defined here
    MPI_Dist_graph_create_adjacent(base.get(), in_degree, sources.get_data(),
                                   MPI_UNWEIGHTED, out_degree,
                                   destinations.get_data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &neighbor_comm.get());

    return neighbor_comm;
}


template <typename ValueType, typename IndexType = int32>
struct overlapping_vector
    : public EnableDistributedLinOp<overlapping_vector<ValueType, IndexType>>,
      public DistributedBase,
      public ::gko::EnableCreateMethod<
          overlapping_vector<ValueType, IndexType>> {
    using value_type = ValueType;
    using index_type = IndexType;
    using local_vector_type = matrix::Dense<value_type>;
    using partition_type = overlapping_partition<index_type>;

    struct interleaved_deleter {
        void operator()(local_vector_type* ptr)
        {
            if (mode == transformation::set) {
                // normal scatter
            }
            if (mode == transformation::add) {
                // scatter with add
            }
            delete ptr;
        }

        std::unique_ptr<local_vector_type> original;
        transformation mode;
    };

    struct blocked_deleter {
        void operator()(local_vector_type* ptr)
        {
            if (mode == transformation::set) {
                // do nothing
            }
            if (mode == transformation::add) {
                original->add_scaled(gko::initialize<local_vector_type>(
                                         {1.0}, original->get_executor()),
                                     ptr);
            }
            delete ptr;
        }

        std::unique_ptr<local_vector_type> original;
        transformation mode;
    };


    size_type get_stride() const;

    size_type get_num_stored_elems() const;

    void make_consistent(transformation mode)
    {
        auto recv_idxs = part_->get_recv_indices();
        auto send_idxs = part_->get_send_indices();

        std::vector<int> send_sizes(send_idxs.target_ids.get_num_elems() +
                                    1);  // first value is ignored
        std::vector<int> send_offsets(send_sizes.size());
        std::vector<int> recv_sizes(recv_idxs.target_ids.get_num_elems() + 1);
        std::vector<int> recv_offsets(recv_sizes.size());

        auto exec = this->get_executor();  // should be exec of part_
        auto host_exec = exec->get_master();
        auto fill_size_offsets = [&](std::vector<int>& sizes,
                                     std::vector<int>& offsets,
                                     const auto& overlap) {
            std::visit(
                overloaded(
                    [&](const typename partition_type::overlap_indices::blocked&
                            idxs) {
                        make_array_view(host_exec, offsets.size(),
                                        offsets.data()) =
                            make_array_view(exec, idxs.get_num_subsets() + 1,
                                            idxs.get_superset_indices());
                        std::adjacent_difference(offsets.begin(), offsets.end(),
                                                 sizes);
                    },
                    [&](const typename partition_type::overlap_indices::
                            interleaved& idxs) {
                        for (int i = 0; i < idxs.size(); ++i) {
                            sizes[i + 1] = idxs[i].get_num_elems();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin());
                    }),
                overlap.idxs);
        };
        fill_size_offsets(send_sizes, send_offsets, send_idxs);
        fill_size_offsets(recv_sizes, recv_offsets, recv_idxs);

        // automatically copies back/adds if necessary
        auto recv_ptr = std::visit(
            [this, mode, idxs = recv_idxs](auto&& arg)
                -> std::unique_ptr<local_vector_type,
                                   std::function<void(local_vector_type*)>> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<
                                  T, typename partition_type::blocked_idxs>) {
                    return {get_overlap_block(arg),
                            blocked_deleter{get_overlap_block(arg), mode}};
                }
                if constexpr (std::is_same_v<
                                  T,
                                  typename partition_type::interleaved_idxs>) {
                    recv_cache_.init(this->executor(),
                                     {this->part_->get_overlap_num_elems(idxs),
                                      this->get_size()[1]});

                    return {make_dense_view(recv_cache_.get()).release(),
                            interleaved_deleter{as_local_vector(), mode}};
                }
            },
            recv_idxs.idxs);
        auto send_ptr = std::visit(
            [this, idxs = send_idxs](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<
                                  T, typename partition_type::blocked_idxs>) {
                    return get_overlap_block(arg);
                }
                if constexpr (std::is_same_v<
                                  T,
                                  typename partition_type::interleaved_idxs>) {
                    send_cache_.init(this->executor(),
                                     {this->part_->get_overlap_num_elems(idxs),
                                      this->get_size()[1]});

                    size_type offset = 0;
                    for (int i = 0; i < arg.idxs.size(); ++i) {
                        // need direct support for index_set
                        auto full_idxs = arg.idxs[i].to_global_indices();
                        as_local_vector()->row_gather(
                            full_idxs,
                            send_cache_->create_submatrix(
                                {offset, offset + full_idxs.get_num_elems()},
                                {0, this->get_size()[1]}));
                    }

                    return make_dense_view(send_cache_.get());
                }
            },
            send_idxs);

        MPI_Neighbor_alltoallv(
            send_ptr->get_values(), send_sizes.data(), send_offsets.data(),
            MPI_DOUBLE, recv_ptr->get_values(), recv_sizes.data(),
            recv_offsets.data(), MPI_DOUBLE, this->get_communicator().get());
    }

    template <typename F, typename = std::enable_if_t<
                              std::is_invocable_v<F, double, double>>>
    void make_consistent(F&& transformation);

    /**
     * could add non-const versions with custom deleter to write back changes
     */
    std::unique_ptr<const local_vector_type> extract_local() const
    {
        auto local_size =
            dim<2>{part_->get_local_indices().get_size(), this->get_size()[1]};
        if (part_->has_grouped_indices()) {
            auto exec = this->get_executor();
            return local_vector_type::create_const(
                exec, local_size, buffer_.as_const_view(), this->get_stride());
        } else {
            // extract / weight vector
        }
    }

    std::unique_ptr<const local_vector_type> extract_non_local() const
    {
        if (part_->has_grouped_indices()) {
            return get_overlap_block(part_->get_recv_indices());
        } else {
            // extract / weight vector
        }
    }

    std::unique_ptr<local_vector_type, std::function<void(local_vector_type*)>>
    extract_writable_non_local()
    {
        if (part_->has_grouped_indices()) {
            return {get_overlap_block(part_->get_recv_indices()).release(),
                    blocked_deleter{nullptr, transformation::set}};
        } else {
            // extract / weight vector
            // use interleaved deleter
        }
    }

private:
    overlapping_vector(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        std::shared_ptr<overlapping_partition<index_type>> part = {})
        : EnableDistributedLinOp<overlapping_vector<ValueType, IndexType>>(
              exec, part_->get_size()),
          DistributedBase(std::move(comm)),
          part_(std::move(part)),
          buffer_(exec, part_->get_size())
    {}

    std::unique_ptr<local_vector_type> as_local_vector()
    {
        return local_vector_type::create(this->get_executor(), this->get_size(),
                                         buffer_.as_view(), this->get_stride());
    }

    std::unique_ptr<const local_vector_type> as_local_vector() const
    {
        return local_vector_type::create_const(
            this->get_executor(), this->get_size(), buffer_.as_const_view(),
            this->get_stride());
    }

    std::unique_ptr<local_vector_type> get_overlap_block(
        const typename partition_type::overlap_indices& idxs)
    {
        const auto& block_idxs =
            std::get<typename partition_type::overlap_indices::blocked>(
                idxs.idxs);
        return as_local_vector()->create_submatrix(
            {this->part_->get_local_indices().get_size(),
             block_idxs.get_size()},
            {0, this->get_size()[1]});
    }

    std::unique_ptr<const local_vector_type> get_overlap_block(
        const typename partition_type::overlap_indices& idxs) const
    {
        const auto& block_idxs =
            std::get<typename partition_type::overlap_indices::blocked>(
                idxs.idxs);
        return as_local_vector()->create_submatrix(
            {this->part_->get_local_indices().get_size(),
             block_idxs.get_size()},
            {0, this->get_size()[1]});
    }


    std::shared_ptr<overlapping_partition<index_type>> part_;
    // contains local+nonlocal values
    // might switch to dense directly
    array<double> buffer_;

    detail::DenseCache<ValueType> recv_cache_;
    detail::DenseCache<ValueType> send_cache_;
};


}  // namespace gko::experimental::distributed

#endif
#endif  // OVERLAPPING_VECTOR_OVERLAPPING_VECTOR_HPP
