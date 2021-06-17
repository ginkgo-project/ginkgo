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

#include "batch_solver.hpp"


int main(int argc, char *argv[])
{
    return read_data_and_launch_benchmark(argc, argv, true);
    /*
// Set the default repetitions = 1.
FLAGS_repetitions = 1;
std::string header =
    "A benchmark for measuring performance of Ginkgo's batch solvers.\n";
std::string format =
    std::string() + "  [\n" +
    "    { \"problem\": \"my_file.mtx\",  \"spmv\": { <matrix format>\" "
    "},\n"
    "    { \"problem\": \"my_file.mtx\",  \"spmv\": { <matrix format>\" "
    "},\n"
    "  ]\n\n" +
    "  \"optimal_format\" can be one of the recognized spmv "
    "format\n\n";
initialize_argument_parsing(&argc, &argv, header, format);

std::stringstream ss_rel_res_goal;
ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

std::string extra_information =
    "Running " + FLAGS_batch_solvers + " with " +
    std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
    ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
    std::to_string(FLAGS_nrhs) + "\nThe number of batch entries is " +
    std::to_string(FLAGS_num_batches) + "\n";
print_general_information(extra_information);

auto exec = get_executor();
auto solvers = split(FLAGS_batch_solvers, ',');

rapidjson::Document test_cases;
rapidjson::IStreamWrapper jcin(std::cin);
test_cases.ParseStream(jcin);

if (!test_cases.IsArray()) {
    if (FLAGS_using_suite_sparse) {
        print_batch_config_error_and_exit();
    } else {
        print_config_error_and_exit();
    }
}

auto engine = get_engine();
auto &allocator = test_cases.GetAllocator();

for (auto &test_case : test_cases.GetArray()) {
    try {
        // set up benchmark
        if (FLAGS_using_suite_sparse) {
            validate_option_object(test_case);
        } else {
            validate_batch_option_object(test_case);
        }
        if (!test_case.HasMember("batch_solver")) {
            test_case.AddMember("batch_solver",
                                rapidjson::Value(rapidjson::kObjectType),
                                allocator);
        }
        auto &solver_case = test_case["batch_solver"];
        if (!FLAGS_overwrite &&
            all_of(begin(solvers), end(solvers),
                   [&solver_case](const std::string &s) {
                       return solver_case.HasMember(s.c_str());
                   })) {
            continue;
        }
        std::clog << "Running test case: " << test_case << std::endl;
        auto nrhs = FLAGS_nrhs;
        auto nbatch = FLAGS_num_batches;
        auto ndup = FLAGS_num_duplications;
        auto data = std::vector<gko::matrix_data<etype>>(nbatch);
        auto scale_data = std::vector<gko::matrix_data<etype>>(nbatch);
        using Vec = gko::matrix::BatchDense<etype>;
        std::shared_ptr<gko::BatchLinOp> system_matrix;
        std::unique_ptr<Vec> b;
        std::unique_ptr<Vec> x;
        std::unique_ptr<Vec> scaling_vec;
        std::string fbase;
        if (FLAGS_using_suite_sparse) {
            GKO_ASSERT(ndup == nbatch);
            fbase = test_case["filename"].GetString();
            std::ifstream mtx_fd(fbase);
            data[0] = gko::read_raw<etype>(mtx_fd);
        } else {
            for (size_type i = 0; i < data.size(); ++i) {
                std::string mat_str;
                if (FLAGS_batch_scaling == "implicit") {
                    mat_str = "A_scaled.mtx";
                } else {
                    mat_str = "A.mtx";
                }
                fbase = std::string(test_case["problem"].GetString()) +
                        "/" + std::to_string(i) + "/";
                std::string fname = fbase + mat_str;
                std::ifstream mtx_fd(fname);
                data[i] = gko::read_raw<etype>(mtx_fd);
                if (FLAGS_batch_scaling == "explicit") {
                    std::string scale_fname = fbase + "S.mtx";
                    std::ifstream scale_fd(scale_fname);
                    scale_data[i] = gko::read_raw<etype>(scale_fd);
                }
            }
        }

        if (FLAGS_using_suite_sparse) {
            system_matrix = share(formats::batch_matrix_factory.at(
                "batch_csr")(exec, ndup, data[0]));
        } else {
            system_matrix = share(formats::batch_matrix_factory2.at(
                "batch_csr")(exec, ndup, data));
            if (FLAGS_batch_scaling == "explicit") {
                auto temp_scaling_op = formats::batch_matrix_factory2.at(
                    "batch_dense")(exec, ndup, scale_data);
                scaling_vec = std::move(std::unique_ptr<Vec>(
                    static_cast<Vec *>(temp_scaling_op.release())));
            }
        }
        if (FLAGS_using_suite_sparse) {
            b = generate_rhs(exec, system_matrix, engine, fbase);
        } else {
            if (FLAGS_rhs_generation == "file") {
                auto b_data = std::vector<gko::matrix_data<etype>>(nbatch);
                for (size_type i = 0; i < data.size(); ++i) {
                    std::string b_str;
                    if (FLAGS_batch_scaling == "implicit") {
                        b_str = "b_scaled.mtx";
                    } else {
                        b_str = "b.mtx";
                    }
                    fbase = std::string(test_case["problem"].GetString()) +
                            "/" + std::to_string(i) + "/";
                    std::string fname = fbase + b_str;
                    std::ifstream mtx_fd(fname);
                    b_data[i] = gko::read_raw<etype>(mtx_fd);
                }
                auto temp_b_op = formats::batch_matrix_factory2.at(
                    "batch_dense")(exec, ndup, b_data);
                b = std::move(std::unique_ptr<Vec>(
                    static_cast<Vec *>(temp_b_op.release())));
            } else {
                b = generate_rhs(exec, system_matrix, engine, fbase);
            }
        }
        x = generate_initial_guess(exec, system_matrix, b.get(), engine);

        std::clog << "Batch Matrix has: "
                  << system_matrix->get_num_batch_entries()
                  << " batches, each of size ("
                  << system_matrix->get_size().at(0)[0] << ", "
                  << system_matrix->get_size().at(0)[1]
                  << ") , with total nnz "
                  << gko::as<gko::matrix::BatchCsr<etype>>(
                         system_matrix.get())
                         ->get_num_stored_elements()
                  << std::endl;

        auto sol_name = begin(solvers);
        for (const auto &solver_name : solvers) {
            std::clog << "\tRunning solver: " << *sol_name << std::endl;
            solve_system(solver_name, exec, system_matrix, lend(b),
                         lend(scaling_vec), lend(x), test_case, allocator);
            backup_results(test_cases);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error setting up solver, what(): " << e.what()
                  << std::endl;
    }
}

std::cout << test_cases << std::endl;
    */
}
