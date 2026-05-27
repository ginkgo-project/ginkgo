// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

#include <ginkgo/ginkgo.hpp>

#include <ginkgo/core/log/logger.hpp>


/**
 * Custom MPI Logger that logs MPI events.
 */
class MpiLogger : public gko::log::Logger {
public:
    MpiLogger(int rank)
        : gko::log::Logger(gko::log::Logger::mpi_events_mask), rank_(rank)
    {}

    void on_mpi_send_started(const gko::Executor* exec, const int& dest_rank,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Sending " << count << " elements ("
                  << bytes << " bytes) to Rank " << dest_rank << "..."
                  << std::endl;
    }

    void on_mpi_send_completed(const gko::Executor* exec, const int& dest_rank,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Send to Rank " << dest_rank
                  << " completed (" << bytes << " bytes)." << std::endl;
    }

    void on_mpi_recv_started(const gko::Executor* exec, const int& src_rank,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Receiving " << count
                  << " elements (" << bytes << " bytes) from Rank " << src_rank
                  << "..." << std::endl;
    }

    void on_mpi_recv_completed(const gko::Executor* exec, const int& src_rank,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Recv from Rank " << src_rank
                  << " completed (" << bytes << " bytes)." << std::endl;
    }

    void on_mpi_all_reduce_started(const gko::Executor* exec,
                                   const gko::size_type& count,
                                   const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllReduce started for " << count
                  << " elements (" << bytes << " bytes)..." << std::endl;
    }

    void on_mpi_all_reduce_completed(const gko::Executor* exec,
                                     const gko::size_type& count,
                                     const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllReduce completed for " << count
                  << " elements (" << bytes << " bytes)." << std::endl;
    }

    void on_mpi_broadcast_started(const gko::Executor* exec,
                                  const int& root_rank,
                                  const gko::size_type& count,
                                  const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Broadcast started from root "
                  << root_rank << " (" << bytes << " bytes)..." << std::endl;
    }

    void on_mpi_broadcast_completed(const gko::Executor* exec,
                                    const int& root_rank,
                                    const gko::size_type& count,
                                    const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Broadcast completed (" << bytes
                  << " bytes)." << std::endl;
    }

    void on_mpi_reduce_started(const gko::Executor* exec, const int& root_rank,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Reduce started to root "
                  << root_rank << " (" << bytes << " bytes)..." << std::endl;
    }

    void on_mpi_reduce_completed(const gko::Executor* exec,
                                 const int& root_rank,
                                 const gko::size_type& count,
                                 const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Reduce completed (" << bytes
                  << " bytes)." << std::endl;
    }

    void on_mpi_gather_started(const gko::Executor* exec, const int& root_rank,
                               const gko::size_type& send_count,
                               const gko::size_type& send_bytes,
                               const gko::size_type& recv_count,
                               const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Gather started to root "
                  << root_rank << "..." << std::endl;
    }

    void on_mpi_gather_completed(
        const gko::Executor* exec, const int& root_rank,
        const gko::size_type& send_count, const gko::size_type& send_bytes,
        const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Gather completed." << std::endl;
    }

    void on_mpi_scatter_started(const gko::Executor* exec, const int& root_rank,
                                const gko::size_type& send_count,
                                const gko::size_type& send_bytes,
                                const gko::size_type& recv_count,
                                const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Scatter started from root "
                  << root_rank << "..." << std::endl;
    }

    void on_mpi_scatter_completed(
        const gko::Executor* exec, const int& root_rank,
        const gko::size_type& send_count, const gko::size_type& send_bytes,
        const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Scatter completed." << std::endl;
    }

    void on_mpi_all_gather_started(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllGather started..." << std::endl;
    }

    void on_mpi_all_gather_completed(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllGather completed." << std::endl;
    }

    void on_mpi_all_to_all_started(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllToAll started..." << std::endl;
    }

    void on_mpi_all_to_all_completed(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] AllToAll completed." << std::endl;
    }

    void on_mpi_scan_started(const gko::Executor* exec,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Scan started (" << bytes
                  << " bytes)..." << std::endl;
    }

    void on_mpi_scan_completed(const gko::Executor* exec,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        std::cout << "[Rank " << rank_ << "] Scan completed." << std::endl;
    }

    void on_mpi_barrier_started(const gko::Executor* exec) const override
    {
        std::cout << "[Rank " << rank_ << "] Barrier started..." << std::endl;
    }

    void on_mpi_barrier_completed(const gko::Executor* exec) const override
    {
        std::cout << "[Rank " << rank_ << "] Barrier completed." << std::endl;
    }

private:
    int rank_;
};


int main(int argc, char* argv[])
{
    const gko::experimental::mpi::environment env(argc, argv);

    using ValueType = double;
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;

    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // Setup custom MPI logger
    auto mpi_logger = std::make_shared<MpiLogger>(rank);
    comm.add_logger(mpi_logger);

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    const auto exec = exec_map.at(executor_string)();

    const gko::size_type num_rows = 16;
    auto partition = gko::share(part_type::build_from_mapping(
        exec,
        gko::array<gko::int32>(
            exec, {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
        4));

    // Distributed vector operation
    auto vec_x = dist_vec::create(
        exec, comm, gko::dim<2>{num_rows, 1},
        gko::dim<2>{static_cast<gko::size_type>(partition->get_part_size(rank)),
                    1});
    auto vec_y = dist_vec::create(
        exec, comm, gko::dim<2>{num_rows, 1},
        gko::dim<2>{static_cast<gko::size_type>(partition->get_part_size(rank)),
                    1});
    vec_x->fill(1.0);
    vec_y->fill(2.0);

    if (rank == 0) {
        std::cout
            << "Starting distributed solver apply (will trigger MPI events)..."
            << std::endl;
    }

    // Create a simple distributed identity matrix
    auto A = gko::share(dist_mtx::create(exec, comm));
    gko::matrix_data<ValueType, GlobalIndexType> A_data{
        gko::dim<2>{num_rows, num_rows}};
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];
    for (int i = range_start; i < range_end; i++) {
        A_data.nonzeros.emplace_back(i, i, 1.0);
    }
    A->read_distributed(A_data, partition);

    auto solver =
        gko::solver::Cg<ValueType>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec))
            .on(exec)
            ->generate(A);


    solver->apply(vec_x.get(), vec_y.get());

    if (rank == 0) {
        std::cout << "Distributed solver apply completed." << std::endl;
        std::cout << "Starting distributed vector dot product (will trigger "
                     "MPI all_reduce)..."
                  << std::endl;
    }

    auto dot_res = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
    vec_x->compute_dot(vec_y.get(), dot_res.get());

    if (rank == 0) {
        std::cout << "Dot product result: " << dot_res->at(0, 0) << std::endl;
        std::cout << "Starting point-to-point communication (send/recv)..."
                  << std::endl;
    }

    // Point-to-point communication
    if (comm.size() >= 2) {
        if (rank == 0) {
            int send_val = 42;
            comm.send(exec, &send_val, 1, 1, 0);

            int i_send_val = 43;
            auto req = comm.i_send(exec, &i_send_val, 1, 1, 1);
            req.wait();
        } else if (rank == 1) {
            int recv_val = 0;
            comm.recv(exec, &recv_val, 1, 0, 0);
            std::cout << "[Rank 1] Received value: " << recv_val << std::endl;

            int i_recv_val = 0;
            auto req = comm.i_recv(exec, &i_recv_val, 1, 0, 1);
            req.wait();
            std::cout << "[Rank 1] Received non-blocking value: " << i_recv_val
                      << std::endl;
        }
    }

    return 0;
}
