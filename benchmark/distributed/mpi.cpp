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


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


// Command-line arguments
DEFINE_string(operations, "SendRecv,Send", "List of operations to benchmark");
DEFINE_string(input, "input.json", "Input json file to read from.");

class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    virtual gko::size_type get_memory() const = 0;
    virtual void prepare(){};
    virtual void run(gko::mpi::communicator comm) = 0;
};


class SendOperation : public BenchmarkOperation {
public:
    SendOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type num_elems)
        : num_elems_(num_elems)
    {
        in_ = std::make_unique<gko::array<etype>>(exec, num_elems_);
        out_ = std::make_unique<gko::array<etype>>(exec, num_elems_);
    }

    void prepare() override
    {
        in_->fill(2);
        out_->fill(3);
    }

    gko::size_type get_memory() const override
    {
        return in_->get_num_elems() * sizeof(etype);
    }

    void run(gko::mpi::communicator comm) override
    {
        if (comm.rank() == 0) {
            comm.send(in_->get_data(), num_elems_, 1, 40);
        } else if (comm.rank() == 1) {
            comm.recv(out_->get_data(), num_elems_, 0, 40);
        }
    }

private:
    gko::size_type num_elems_;
    std::unique_ptr<gko::array<etype>> in_;
    std::unique_ptr<gko::array<etype>> out_;
};


class SendRecvOperation : public BenchmarkOperation {
public:
    SendRecvOperation(std::shared_ptr<const gko::Executor> exec,
                      gko::size_type num_elems)
        : num_elems_(num_elems), exec_(exec)
    {
        in_ = std::make_unique<gko::array<etype>>(exec, num_elems_);
        out_ = std::make_unique<gko::array<etype>>(exec, num_elems_);
    }

    void prepare() override
    {
        in_->fill(2);
        out_->fill(3);
    }

    gko::size_type get_memory() const override
    {
        return in_->get_num_elems() * sizeof(etype) * 2;
    }

    void run(gko::mpi::communicator comm) override
    {
        auto req = std::vector<gko::mpi::request>(
            comm.size(), gko::mpi::request{exec_->get_master()});
        auto req1 = std::vector<gko::mpi::request>(
            comm.size(), gko::mpi::request{exec_->get_master()});
        if (comm.rank() == 0) {
            req[0] = comm.i_send(in_->get_data(), num_elems_, 1, 40);
            req1[0] = comm.i_recv(out_->get_data(), num_elems_, 1, 41);
        } else if (comm.rank() == 1) {
            req[1] = comm.i_recv(out_->get_data(), num_elems_, 0, 40);
            req1[1] = comm.i_send(in_->get_data(), num_elems_, 0, 41);
        }
        auto stat1 = gko::mpi::wait_all(req);
        auto stat2 = gko::mpi::wait_all(req1);
    }

private:
    std::shared_ptr<const gko::Executor> exec_;
    gko::size_type num_elems_;
    std::unique_ptr<gko::array<etype>> in_;
    std::unique_ptr<gko::array<etype>> out_;
};


gko::size_type parse_num_elems(rapidjson::Value& test_case)
{
    gko::size_type result;
    result = test_case["n"].GetInt64();
    return result;
}


std::map<std::string,
         std::function<std::unique_ptr<BenchmarkOperation>(
             std::shared_ptr<const gko::Executor>, gko::size_type)>>
    operation_map{{"SendRecv",
                   [](std::shared_ptr<const gko::Executor> exec,
                      gko::size_type num_elems) {
                       return std::make_unique<SendRecvOperation>(exec,
                                                                  num_elems);
                   }},
                  {"Send", [](std::shared_ptr<const gko::Executor> exec,
                              gko::size_type num_elems) {
                       return std::make_unique<SendOperation>(exec, num_elems);
                   }}};


void mpi_bench(const char* operation_name, std::shared_ptr<gko::Executor> exec,
               gko::mpi::communicator comm, rapidjson::Value& test_case,
               rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& mpi_case = test_case["mpi"];
        add_or_set_member(mpi_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op =
            operation_map[operation_name](exec, parse_num_elems(test_case));

        auto timer = get_timer(exec, FLAGS_gpu_timer);
        IterationControl ic(timer);

        // warm run
        for (auto _ : ic.warmup_run()) {
            op->prepare();
            exec->synchronize();
            op->run(comm);
            exec->synchronize();
        }

        // timed run
        op->prepare();
        for (auto _ : ic.run()) {
            op->run(comm);
        }
        const auto runtime = ic.compute_average_time();
        const auto mem = static_cast<double>(op->get_memory());
        const auto repetitions = ic.get_num_repetitions();
        add_or_set_member(mpi_case[operation_name], "time", runtime, allocator);
        add_or_set_member(mpi_case[operation_name], "memory", mem, allocator);
        add_or_set_member(mpi_case[operation_name], "bandwidth", mem / runtime,
                          allocator);
        add_or_set_member(mpi_case[operation_name], "repetitions", repetitions,
                          allocator);

        // compute and write benchmark data
        add_or_set_member(mpi_case[operation_name], "completed", true,
                          allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["mpi"][operation_name], "completed", false,
                          allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case["mpi"][operation_name], "error",
                              msg_value, allocator);
        }
        if (comm.rank() == 0) {
            std::cerr << "Error when processing test case " << test_case << "\n"
                      << "what(): " << e.what() << std::endl;
        }
    }
}


int main(int argc, char* argv[])
{
    gko::mpi::environment mpi_env{argc, argv};

    std::string header =
        "A benchmark for measuring performance of MPI "
        "operations.\nParameters for a benchmark case are:\n"
        "    n: number of rows for vectors output (required)\n"
        "    r: number of columns for vectors (optional, default 1)\n";
    std::string format =
        std::string() + "  [\n    { \"n\": 100 },\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(MPI_COMM_WORLD);

    gko::mpi::communicator comm(MPI_COMM_WORLD, exec);
    const auto rank = comm.rank();
    if (rank == 0) {
        std::string extra_information =
            "The operations are " + FLAGS_operations + "\n";
        print_general_information(extra_information);
    }

    auto operations = split(FLAGS_operations, ',');

    std::ifstream ifs(FLAGS_input);
    rapidjson::IStreamWrapper jcin(ifs);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        std::cerr
            << "Input has to be a JSON array of benchmark configurations:\n"
            << format;
        std::exit(1);
    }

    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            if (!test_case.HasMember("mpi")) {
                test_case.AddMember(
                    "mpi", rapidjson::Value(rapidjson::kObjectType), allocator);
            }
            auto& mpi_case = test_case["mpi"];
            if (!FLAGS_overwrite &&
                all_of(begin(operations), end(operations),
                       [&mpi_case](const std::string& s) {
                           return mpi_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            if (rank == 0) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            for (const auto& operation_name : operations) {
                mpi_bench(operation_name.c_str(), exec, comm, test_case,
                          allocator);
                if (rank == 0) {
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Error setting up benchmark, what(): " << e.what()
                          << std::endl;
            }
        }
    }

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
