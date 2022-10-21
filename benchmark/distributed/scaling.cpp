/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <iostream>
#include <set>
#include <string>

#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/stencil_matrix.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_uint32(nrhs, 1, "The number of right hand sides");


template <typename ValueType>
using dist_vec = gko::experimental::distributed::Vector<ValueType>;
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
using dist_mtx =
    gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                           GlobalIndexType>;


std::string example_config = R"(
  [
    {"size": 100, "stencil": "7pt", "comm_pattern": "optimal",
     "format" : {"local": "csr", "non_local": "coo"}},
    {"filename": "my_file.mtx", "format" : {"local": "ell"}}
  ]
)";


// input validation
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << example_config << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() ||
        !((value.HasMember("size") && value.HasMember("stencil")) ||
          value.HasMember("filename")) ||
        (value.HasMember("format") && !value["format"].IsObject())) {
        print_config_error_and_exit();
    }
}


std::unique_ptr<dist_mtx<etype, itype, gko::int64>> create_distributed_matrix(
    std::shared_ptr<gko::Executor> exec, gko::mpi::communicator comm,
    const std::string& format_local, const std::string& format_non_local,
    const gko::matrix_data<etype, gko::int64>& data,
    const gko::distributed::Partition<itype, gko::int64>* part,
    rapidjson::Value& spmv_case, rapidjson::MemoryPoolAllocator<>& allocator)
{
    auto local_mat = formats::matrix_type_factory.at(format_local)(exec);
    auto non_local_mat =
        formats::matrix_type_factory.at(format_non_local)(exec);

    auto storage_logger = std::make_shared<StorageLogger>();
    exec->add_logger(storage_logger);

    auto dist_mat = dist_mtx<etype, itype, gko::int64>::create(
        exec, comm, local_mat.get(), non_local_mat.get());
    dist_mat->read_distributed(data, part);

    exec->remove_logger(gko::lend(storage_logger));
    storage_logger->write_data(comm, spmv_case, allocator);

    return dist_mat;
}


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void apply_spmv(std::shared_ptr<gko::Executor> exec,
                const dist_mtx<etype, itype, gko::int64>* system_matrix,
                const dist_vec<etype>* b, const dist_vec<etype>* x,
                rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    auto& spmv_case = test_case["spmv"];
    try {
        auto comm = system_matrix->get_communicator();
        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};
        // warm run
        for (auto _ : ic.warmup_run()) {
            auto x_clone = clone(x);
            exec->synchronize();
            comm.synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        // timed run
        auto x_clone = clone(x);
        for (auto _ : ic.run()) {
            comm.synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
        }
        add_or_set_member(spmv_case, "time", ic.compute_average_time(),
                          allocator);
        add_or_set_member(spmv_case, "repetitions", ic.get_num_repetitions(),
                          allocator);

        // compute and write benchmark data
        add_or_set_member(spmv_case, "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(spmv_case, "completed", false, allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(spmv_case, "error", msg_value, allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_matrix_data(
    rapidjson::Value& test_case, gko::mpi::communicator comm)
{
    if (test_case.HasMember("filename")) {
        std::ifstream in(test_case["filename"].GetString());
        return gko::read_generic_raw<ValueType, IndexType>(in);
    } else {
        return generate_stencil<ValueType, IndexType>(
            test_case["stencil"].GetString(), std::move(comm),
            test_case["size"].GetInt64(),
            test_case["comm_pattern"].GetString() == std::string("optimal"));
    }
}


int main(int argc, char* argv[])
{
    gko::mpi::environment mpi_env{argc, argv};

    std::string header =
        "A benchmark for measuring the strong or weak scaling of Ginkgo's "
        "distributed SpMV\n";
    std::string format = example_config + R"(
  The matrix will either be read from an input file if the filename parameter
  is given, or generated as a stencil matrix.
  If the filename parameter is given, all processes will read the file and
  then the matrix is distributed row-block-wise.
  In the other case, a size and stencil parameter have to be provided.
  The size parameter denotes the size per process. It might be adjusted to
  fit the dimensionality of the stencil more easily.
  Possible values for "stencil" are:  5pt (2D), 7pt (3D), 9pt (2D), 27pt (3D).
  Optional values for "comm_pattern" are: stencil, optimal.
  Optional values for "local" and "non_local" are any of the recognized spmv
  formats.
)";
    initialize_argument_parsing(&argc, &argv, header, format);

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::string extra_information;
    if (FLAGS_repetitions == "auto") {
        extra_information =
            "WARNING: repetitions = 'auto' not supported for MPI "
            "benchmarks, setting repetitions to the default value.";
        FLAGS_repetitions = "10";
    }
    if (rank == 0) {
        print_general_information(extra_information);
    }

    std::string json_input;
    if (rank == 0) {
        std::string line;
        while (std::cin >> line) {
            json_input += line;
        }
    }
    auto input_size = json_input.size();
    comm.broadcast(exec->get_master(), &input_size, 1, 0);
    json_input.resize(input_size);
    comm.broadcast(exec->get_master(), &json_input[0],
                   static_cast<int>(input_size), 0);

    rapidjson::Document test_cases;
    test_cases.Parse(json_input.c_str());
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto engine = get_engine();
    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            if (test_case.HasMember("stencil") &&
                !test_case.HasMember("comm_pattern")) {
                add_or_set_member(test_case, "comm_pattern", "optimal",
                                  allocator);
            }
            if (!test_case.HasMember("format")) {
                test_case.AddMember("format",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            if (!test_case["format"].HasMember("local")) {
                add_or_set_member(test_case["format"], "local", "csr",
                                  allocator);
            }
            if (!test_case["format"].HasMember("non_local")) {
                add_or_set_member(test_case["format"], "non_local", "csr",
                                  allocator);
            }
            auto& spmv_case = test_case["spmv"];
            if (rank == 0) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            auto data =
                generate_matrix_data<etype, gko::int64>(test_case, comm);
            auto part = gko::distributed::Partition<itype, gko::int64>::
                build_from_global_size_uniform(
                    exec, comm.size(), static_cast<gko::int64>(data.size[0]));
            const auto global_size = part->get_size();
            const auto local_size =
                static_cast<gko::size_type>(part->get_part_size(rank));

            auto system_matrix = create_distributed_matrix(
                exec, comm, test_case["format"]["local"].GetString(),
                test_case["format"]["non_local"].GetString(), data, part.get(),
                spmv_case, allocator);

            using Vec = dist_vec<etype>;
            auto nrhs = FLAGS_nrhs;
            auto b =
                Vec::create(exec, comm, gko::dim<2>{global_size, nrhs},
                            create_matrix<etype>(
                                exec, gko::dim<2>{local_size, nrhs}, engine)
                                .get());
            auto x = Vec::create(
                exec, comm, gko::dim<2>{global_size, nrhs},
                create_matrix<etype>(exec, gko::dim<2>{local_size, nrhs}, 1)
                    .get());

            if (rank == 0) {
                std::clog << "Running on " << comm.size() << " processes."
                          << std::endl;
                std::clog << "Matrix is of size ("
                          << system_matrix->get_size()[0] << ", "
                          << system_matrix->get_size()[1] << ")." << std::endl;
            }
            add_or_set_member(spmv_case, "num_procs", comm.size(), allocator);
            add_or_set_member(spmv_case, "global_size", global_size, allocator);
            add_or_set_member(spmv_case, "local_size", local_size, allocator);
            add_or_set_member(spmv_case, "num_rhs", FLAGS_nrhs, allocator);
            apply_spmv(exec, system_matrix.get(), b.get(), x.get(), test_case,
                       allocator);
            if (rank == 0) {
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up spmv, what(): " << e.what()
                      << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
