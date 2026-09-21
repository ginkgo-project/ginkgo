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
    }

    return 0;
}
