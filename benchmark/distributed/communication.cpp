/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <ginkgo/ginkgo.hpp>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "core/test/utils/matrix_generator.hpp"

DEFINE_string(input, "input.json",
              "The name of the JSON file containing the run configuration");
DEFINE_string(operations,
              "read-mat,gather-vec,gather-mat,convert-vec,convert-mat", "");


template <typename ValueType>
using dist_vec = gko::distributed::Vector<ValueType, gko::int32>;
template <typename ValueType>
using dist_mat = gko::distributed::Matrix<ValueType, gko::int32>;
template <typename ValueType>
using mat = gko::matrix::Csr<etype, gko::int32>;


gko::matrix_data<etype, gko::int64> create_27pt_pattern_data(
    const gko::size_type start_row, const gko::size_type end_row,
    const gko::size_type global_n)
{
    const auto dp =
        static_cast<gko::size_type>(std::floor(std::pow(global_n, 1 / 3.)));
    gko::matrix_data<etype, gko::int64> data{gko::dim<2>{global_n, global_n}};
    for (gko::size_type row = start_row; row < end_row; ++row) {
        for (int nz : {-1, 0, 1}) {
            for (int ny : {-1, 0, 1}) {
                for (int nx : {-1, 0, 1}) {
                    const auto col = row + nx + ny * dp + nz * dp * dp;
                    if (col >= 0 && col < global_n) {
                        data.nonzeros.emplace_back(row, col, 1.);
                    }
                }
            }
        }
    }
    return data;
}


std::unique_ptr<dist_mat<etype>> create_27pt_pattern(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<gko::mpi::communicator> comm,
    std::shared_ptr<const gko::distributed::Partition<gko::int32>> part,
    gko::size_type local_n)
{
    const auto rank = comm->rank();
    const auto global_n = local_n * comm->size();
    auto data = create_27pt_pattern_data(local_n * rank, local_n * (rank + 1),
                                         global_n);
    auto mat = dist_mat<etype>::create(exec, comm);
    mat->read_distributed(data, part);
    return mat;
}


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    virtual void prepare(){};
    virtual void run() = 0;
};


class GatherOperation : public BenchmarkOperation {
public:
    GatherOperation(std::shared_ptr<const gko::Executor> exec,
                    std::shared_ptr<gko::mpi::communicator> from_comm,
                    gko::size_type n)
        : repartitioner(gko::distributed::Repartitioner<gko::int32>::create(
              from_comm,
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, from_comm->size(), n * from_comm->size())),
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, 1, n * from_comm->size())))),
          from_vec(gko::distributed::Vector<etype>::create(
              exec, from_comm, repartitioner->get_from_partition(),
              gko::dim<2>{n * from_comm->size(), 1}, gko::dim<2>{n, 1})),
          to_vec(gko::distributed::Vector<etype>::create(
              exec, from_comm, repartitioner->get_to_partition()))
    {
        create_matrix(exec, from_vec->get_local()->get_size(),
                      gko::one<etype>())
            ->move_to(from_vec->get_local());
    }

    void prepare() override
    {
        gko::distributed::Vector<etype>::create(
            to_vec->get_executor(), to_vec->get_communicator(),
            repartitioner->get_to_partition())
            ->move_to(gko::lend(to_vec));
    }

    void run() override
    {
        repartitioner->gather(gko::lend(from_vec), gko::lend(to_vec));
    }

private:
    std::shared_ptr<gko::distributed::Repartitioner<gko::int32>> repartitioner;
    std::unique_ptr<dist_vec<etype>> from_vec;
    std::unique_ptr<dist_vec<etype>> to_vec;
};


class ConvertOperation : public BenchmarkOperation {
public:
    ConvertOperation(std::shared_ptr<const gko::Executor> exec,
                     std::shared_ptr<gko::mpi::communicator> from_comm,
                     gko::size_type n)
        : from_vec(dist_vec<etype>::create(
              exec, from_comm,
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, from_comm->size(), n * from_comm->size())),
              gko::dim<2>{n * from_comm->size(), 1}, gko::dim<2>{n, 1})),
          to_vec(vec<etype>::create(exec))
    {
        create_matrix(exec, from_vec->get_local()->get_size(),
                      gko::one<etype>())
            ->move_to(from_vec->get_local());
    }

    void prepare() override
    {
        vec<etype>::create(to_vec->get_executor())->move_to(gko::lend(to_vec));
    }

    void run() override { from_vec->convert_to(gko::lend(to_vec)); }

private:
    std::unique_ptr<dist_vec<etype>> from_vec;
    std::unique_ptr<vec<etype>> to_vec;
};


class GatherMatrixOperation : public BenchmarkOperation {
public:
    GatherMatrixOperation(std::shared_ptr<const gko::Executor> exec,
                          std::shared_ptr<gko::mpi::communicator> from_comm,
                          gko::size_type n)
        : repartitioner(gko::distributed::Repartitioner<gko::int32>::create(
              from_comm,
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, from_comm->size(), n * from_comm->size())),
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, 1, n * from_comm->size())))),
          from_mat(create_27pt_pattern(exec, from_comm,
                                       repartitioner->get_from_partition(), n)),
          to_mat(dist_mat<etype>::create(exec,
                                         repartitioner->get_to_communicator()))
    {}

    void prepare() override
    {
        dist_mat<etype>::create(to_mat->get_executor(),
                                to_mat->get_communicator())
            ->move_to(gko::lend(to_mat));
    }

    void run() override
    {
        repartitioner->gather(gko::lend(from_mat), gko::lend(to_mat));
    }

private:
    std::shared_ptr<gko::distributed::Repartitioner<gko::int32>> repartitioner;
    std::unique_ptr<dist_mat<etype>> from_mat;
    std::unique_ptr<dist_mat<etype>> to_mat;
};


class ConvertMatrixOperation : public BenchmarkOperation {
public:
    ConvertMatrixOperation(std::shared_ptr<const gko::Executor> exec,
                           std::shared_ptr<gko::mpi::communicator> from_comm,
                           gko::size_type n)
        : from_mat(create_27pt_pattern(
              exec, from_comm,
              gko::share(
                  gko::distributed::Partition<gko::int32>::build_uniformly(
                      exec, from_comm->size(), n * from_comm->size())),
              n)),
          to_mat(mat<etype>::create(exec))
    {}

    void prepare() override
    {
        mat<etype>::create(to_mat->get_executor())->move_to(gko::lend(to_mat));
    }

    void run() override { from_mat->convert_to(gko::lend(to_mat)); }

private:
    std::unique_ptr<dist_mat<etype>> from_mat;
    std::unique_ptr<mat<etype>> to_mat;
};


class ReadDistributedOperation : public BenchmarkOperation {
public:
    ReadDistributedOperation(std::shared_ptr<const gko::Executor> exec,
                             std::shared_ptr<gko::mpi::communicator> comm,
                             gko::size_type n)
        : data(create_27pt_pattern_data(
              n * comm->rank(), n * (comm->rank() + 1), n * comm->size())),
          part(gko::share(
              gko::distributed::Partition<gko::int32>::build_uniformly(
                  exec, comm->size(), n * comm->size()))),
          read_mat(dist_mat<etype>::create(exec, comm))
    {}

    void prepare() override
    {
        dist_mat<etype>::create(read_mat->get_executor(),
                                read_mat->get_communicator())
            ->move_to(gko::lend(read_mat));
    }

    void run() override { read_mat->read_distributed(data, part); }

private:
    gko::matrix_data<etype, gko::int64> data;
    std::shared_ptr<gko::distributed::Partition<gko::int32>> part;
    std::unique_ptr<dist_mat<etype>> read_mat;
};


std::map<std::string,
         std::function<std::unique_ptr<BenchmarkOperation>(
             std::shared_ptr<const gko::Executor>,
             std::shared_ptr<gko::mpi::communicator>, gko::size_type)>>
    operation_map{
        {"read-mat",
         [](std::shared_ptr<const gko::Executor> exec,
            std::shared_ptr<gko::mpi::communicator> from_comm,
            gko::size_type local_n) {
             return std::make_unique<ReadDistributedOperation>(exec, from_comm,
                                                               local_n);
         }},
        {"gather-vec",
         [](std::shared_ptr<const gko::Executor> exec,
            std::shared_ptr<gko::mpi::communicator> from_comm,
            gko::size_type local_n) {
             return std::make_unique<GatherOperation>(exec, from_comm, local_n);
         }},
        {"convert-vec",
         [](std::shared_ptr<const gko::Executor> exec,
            std::shared_ptr<gko::mpi::communicator> from_comm,
            gko::size_type local_n) {
             return std::make_unique<ConvertOperation>(exec, from_comm,
                                                       local_n);
         }},
        {"gather-mat",
         [](std::shared_ptr<const gko::Executor> exec,
            std::shared_ptr<gko::mpi::communicator> from_comm,
            gko::size_type local_n) {
             return std::make_unique<GatherMatrixOperation>(exec, from_comm,
                                                            local_n);
         }},
        {"convert-mat",
         [](std::shared_ptr<const gko::Executor> exec,
            std::shared_ptr<gko::mpi::communicator> from_comm,
            gko::size_type local_n) {
             return std::make_unique<ConvertMatrixOperation>(exec, from_comm,
                                                             local_n);
         }},
    };


void apply_operation(const char* operation_name,
                     std::shared_ptr<const gko::Executor> exec,
                     std::shared_ptr<gko::mpi::communicator> from_comm,
                     gko::size_type local_n, rapidjson::Value& test_case,
                     rapidjson::MemoryPoolAllocator<>& allocator)
{
    add_or_set_member(test_case, operation_name,
                      rapidjson::Value(rapidjson::kObjectType), allocator);

    auto op = operation_map[operation_name](exec, from_comm, local_n);

    IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};
    auto it_logger = std::make_shared<IterationLogger>(exec);
    for (auto _ : ic.warmup_run()) {
        op->prepare();
        exec->synchronize();
        op->run();
        exec->synchronize();
    }

    op->prepare();
    MPI_Barrier(from_comm->get());
    for (auto _ : ic.run()) {
        op->run();
        exec->synchronize();
    }
    auto time = ic.compute_average_time();
    gko::mpi::all_reduce(&time, 1, gko::mpi::op_type::sum, from_comm);
    if (from_comm->rank() == 0) {
        auto& op_case = test_case[operation_name];
        add_or_set_member(op_case, "time", time / from_comm->size(), allocator);
        add_or_set_member(op_case, "repetitions", ic.get_num_repetitions(),
                          allocator);
    }
}


int main(int argc, char* argv[])
{
    auto mpi_guard = gko::mpi::init_finalize(argc, argv);

    auto comm_world = gko::mpi::communicator::create_world();
    auto rank = comm_world->rank();

    std::string header =
        "A benchmark for measuring performance of Ginkgo's MPI "
        "communications.\nParameters for a benchmark case are:\n"
        "    n: number of rows for vectors and (square) matrix output "
        "(required)\n";
    std::string format =
        std::string() + "  [\n    { \"n\": 100 },\n    { \"n\": 200}\n]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    if (rank == 0) {
        std::string extra_information;
        print_general_information(extra_information);
    }

    auto exec = executor_factory.at(FLAGS_executor)();

    auto operations = split(FLAGS_operations, ',');

    std::ifstream input_file(FLAGS_input);
    rapidjson::IStreamWrapper jcin(input_file);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    bool is_array = test_cases.IsArray();
    gko::mpi::all_reduce(&is_array, 1, gko::mpi::op_type::logical_and,
                         comm_world);
    if (!is_array) {
        if (rank == 0) {
            std::cerr
                << "Input has to be a JSON array of benchmark configurations:\n"
                << format;
        }
        std::exit(1);
    }

    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            if (rank == 0) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            for (const auto& operation_name : operations) {
                auto local_n = test_case["n"].GetInt64();

                apply_operation(operation_name.c_str(), exec, comm_world,
                                local_n, test_case, allocator);
                if (rank == 0) {
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                    backup_results(test_cases);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up benchmark, what(): " << e.what()
                      << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
