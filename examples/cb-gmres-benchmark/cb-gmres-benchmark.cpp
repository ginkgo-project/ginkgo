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

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <libpressio_ext/cpp/json.h>
#include <libpressio_ext/cpp/libpressio.h>
#include <libpressio_meta.h>
#include <nlohmann/json.hpp>


struct user_launch_parameter {
    std::string exec_string{"reference"};
    std::string matrix_path{"data/A.mtx"};
    std::string compression_json_file{"lp_configs"};
    unsigned stop_iter{101};
    double stop_rel_res_norm{1e-16};
    unsigned jacobi_bs{0};
    unsigned krylov_dim{100};
};

struct solver_settings {
    unsigned krylov_dim;
    unsigned stop_iter;
    double stop_rel_res;
    gko::solver::cb_gmres::storage_precision storage_prec;
    std::shared_ptr<const gko::LinOp> precond;
    std::function<void(void*)> init_compressor;
};

template <typename T>
struct solver_result {
    unsigned iters;
    double time_s;
    T init_res_norm;
    T res_norm;
    const std::vector<T>* residual_norm_history;
};


/**
 * Logs both the residual norm history and the iteration count.
 * Note: the residual norm history logs only the norm of the first vector!
 */
template <typename ValueType>
class ConvergenceHistoryLogger : public gko::log::Logger {
public:
    using RealValueType = gko::remove_complex<ValueType>;

    void on_criterion_check_completed(
        const gko::stop::Criterion* criterion,
        const gko::size_type& num_iterations, const gko::LinOp* residual,
        const gko::LinOp* residual_norm, const gko::LinOp* solution,
        const gko::uint8& stopping_id, const bool& set_finalized,
        const gko::array<gko::stopping_status>* status, const bool& one_changed,
        const bool& all_converged) const override
    {
        this->on_criterion_check_completed(criterion, num_iterations, residual,
                                           residual_norm, nullptr, solution,
                                           stopping_id, set_finalized, status,
                                           one_changed, all_converged);
    }

    void on_criterion_check_completed(
        const gko::stop::Criterion* criterion,
        const gko::size_type& num_iterations, const gko::LinOp* residual,
        const gko::LinOp* residual_norm, const gko::LinOp* implicit_sq_resnorm,
        const gko::LinOp* solution, const gko::uint8& stopping_id,
        const bool& set_finalized,
        const gko::array<gko::stopping_status>* status, const bool& one_changed,
        const bool& all_converged) const override
    {
        num_iterations_ = num_iterations;
        residual_norm_history_.push_back(
            residual_norm->get_executor()->copy_val_to_host(
                reinterpret_cast<const gko::matrix::Dense<RealValueType>*>(
                    residual_norm)
                    ->get_const_values()));
    }

    /**
     * Creates a convergence logger. This dynamically allocates the memory,
     * constructs the object and returns an std::unique_ptr to this object.
     *
     * @return an std::unique_ptr to the the constructed object
     */
    static std::unique_ptr<ConvergenceHistoryLogger> create()
    {
        return std::unique_ptr<ConvergenceHistoryLogger>(
            new ConvergenceHistoryLogger());
    }

    void reset()
    {
        num_iterations_ = 0;
        residual_norm_history_.clear();
    }

    std::size_t get_num_iterations() const { return num_iterations_; }

    const std::vector<RealValueType>& get_residual_norm_history() const
    {
        return residual_norm_history_;
    }

protected:
    /**
     * Creates a Convergence logger.
     */
    explicit ConvergenceHistoryLogger()
        : gko::log::Logger(gko::log::Logger::criterion_check_completed_mask),
          residual_norm_history_{}
    {
        residual_norm_history_.reserve(21000);
    }

private:
    mutable std::size_t num_iterations_{};
    mutable std::vector<RealValueType> residual_norm_history_;
};


template <typename ValueType, typename IndexType = int>
class Benchmark {
private:
    using RealValueType = gko::remove_complex<ValueType>;
    using Vector = gko::matrix::Dense<ValueType>;
    using NormVector = gko::matrix::Dense<RealValueType>;
    static constexpr int repeats{1};

public:
    Benchmark(std::shared_ptr<const gko::Executor> exec,
              std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> mtx,
              std::unique_ptr<Vector> init_x, std::unique_ptr<Vector> rhs)
        : exec_{std::move(exec)},
          mtx_{std::move(mtx)},
          init_x_{std::move(init_x)},
          rhs_{std::move(rhs)},
          one_{gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec_)},
          neg_one_{
              gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec_)},
          x_{Vector::create(exec_, init_x_->get_size())},
          residual_{Vector::create(exec_, rhs_->get_size())},
          res_norm_{gko::initialize<NormVector>({0.0}, exec_)},
          convergence_history_logger_{
              ConvergenceHistoryLogger<ValueType>::create()}
    {
        x_->copy_from(init_x_.get());
        rhs_->compute_norm2(res_norm_);
        rhs_norm_ = exec_->copy_val_to_host(res_norm_->get_const_values());
        this->compute_residual_norm();
    }

    // Helper function which measures the time of `solver->apply(rhs, x)` in
    // seconds To get an accurate result, the solve is repeated multiple
    // times (while ensuring the initial guess is always the same). The
    // result of the solve will be written to x.
    // Note: Finish processing the residual_norm_history before the destructor
    //       of this object is called! Otherwise, you read undefined memory!
    solver_result<RealValueType> benchmark_solver(solver_settings s_s)
    {
        double duration{0};
        solver_result<RealValueType> result{};

        this->reset();

        // Reset x to the initial guess
        result.init_res_norm = this->compute_residual_norm();

        auto iter_stop = gko::share(gko::stop::Iteration::build()
                                        .with_max_iters(s_s.stop_iter)
                                        .on(exec_));
        auto tol_stop =
            gko::share(gko::stop::ResidualNorm<ValueType>::build()
                           .with_reduction_factor(
                               static_cast<RealValueType>(s_s.stop_rel_res))
                           .with_baseline(gko::stop::mode::rhs_norm)
                           .on(exec_));
        // std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        //    gko::log::Convergence<ValueType>::create();
        // iter_stop->add_logger(logger);
        // tol_stop->add_logger(logger);
        iter_stop->add_logger(convergence_history_logger_);

        // Create solver:
        auto solver_gen = gko::solver::CbGmres<ValueType>::build()
                              .with_criteria(iter_stop, tol_stop)
                              .with_krylov_dim(s_s.krylov_dim)
                              .with_storage_precision(s_s.storage_prec)
                              .with_generated_preconditioner(s_s.precond)
                              .with_init_compressor(s_s.init_compressor)
                              .on(exec_);

        // Generate the actual solver from the factory and the matrix.
        auto solver = solver_gen->generate(mtx_);

        for (int i = 0; i < repeats; ++i) {
            // No need to copy it in the first iteration
            if (i != 0) {
                x_->copy_from(init_x_.get());
            }
            // Make sure all previous executor operations have finished before
            // starting the time
            exec_->synchronize();
            auto tic = std::chrono::steady_clock::now();
            solver->apply(rhs_, x_);
            // Make sure all computations are done before stopping the time
            exec_->synchronize();
            auto tac = std::chrono::steady_clock::now();
            duration += std::chrono::duration<double>(tac - tic).count();
        }

        result.iters = convergence_history_logger_->get_num_iterations();
        result.time_s = duration / static_cast<double>(repeats);
        result.res_norm = this->compute_residual_norm();
        result.residual_norm_history =
            &convergence_history_logger_->get_residual_norm_history();
        return result;
    }

    RealValueType get_rhs_norm() const { return rhs_norm_; }

private:
    RealValueType compute_residual_norm()
    {
        residual_->copy_from(rhs_);
        mtx_->apply(one_, x_, neg_one_, residual_);
        residual_->compute_norm2(res_norm_);
        res_norm_value_ =
            exec_->copy_val_to_host(res_norm_->get_const_values());
        return res_norm_value_;
    }

    void reset()
    {
        x_->copy_from(init_x_.get());
        convergence_history_logger_->reset();
    }

    std::shared_ptr<const gko::Executor> exec_;
    std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>> mtx_;
    std::unique_ptr<const Vector> init_x_;
    std::unique_ptr<const Vector> rhs_;
    std::unique_ptr<const Vector> one_;
    std::unique_ptr<const Vector> neg_one_;
    RealValueType rhs_norm_;
    std::unique_ptr<Vector> x_;
    std::unique_ptr<Vector> residual_;
    std::unique_ptr<NormVector> res_norm_;
    RealValueType res_norm_value_;
    std::shared_ptr<ConvergenceHistoryLogger<ValueType>>
        convergence_history_logger_;
};


template <typename ValueType, typename IndexType>
void run_benchmarks(const user_launch_parameter& launch_param)
{
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx_t = gko::matrix::Csr<ValueType, IndexType>;
    using cb_gmres = gko::solver::CbGmres<ValueType>;

    // Map which generates the appropriate executor
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    auto exec = exec_map.at(launch_param.exec_string)();  // throws if not valid

    constexpr char delim = ';';
    constexpr char res_norm_history_delim = '&';
    constexpr char kv_delim = ':';
    const std::string data_begin = "{\n";
    const std::string data_end = "}\n";
    const std::string file_name_info_separator = "__";

    auto mtx =
        share(gko::read<mtx_t>(std::ifstream(launch_param.matrix_path), exec));

    const auto mtx_size = mtx->get_size();
    auto rhs = vec::create(exec, gko::dim<2>{mtx_size[0], 1});
    auto init_x = vec::create(exec, gko::dim<2>{mtx_size[1], 1});

    double rhs_norm{};

    {  // Prepare values and delete all temporaries afterwards
        auto res_host =
            vec::create(exec->get_master(), gko::dim<2>{mtx_size[1], 1});
        ValueType tmp_norm{};
        for (gko::size_type i = 0; i < res_host->get_size()[0]; ++i) {
            const auto val = std::sin(static_cast<ValueType>(i));
            res_host->at(i, 0) = val;
            tmp_norm += val * val;
        }
        tmp_norm = std::sqrt(tmp_norm);
        for (gko::size_type i = 0; i < res_host->get_size()[0]; ++i) {
            res_host->at(i, 0) /= tmp_norm;
        }
        // Write out the actual RHS b
        A->apply(res_host, b);
        auto b_host_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
        b->compute_norm2(b_host_norm);

        rhs_norm = b_host_norm->at(0, 0);

        // As an initial guess, use the right-hand side
        auto init_x_host = clone(exec->get_master(), init_x);
        for (gko::size_type i = 0; i < init_x_host->get_size()[0]; ++i) {
            init_x_host->at(i, 0) = 0;
        }
        init_x->copy_from(init_x_host);
    }

    Benchmark b_object(exec, mtx, std::move(init_x), std::move(rhs));

    using precond_type = gko::preconditioner::Jacobi<ValueType, IndexType>;
    // Default_settings
    solver_settings default_ss{};
    default_ss.stop_iter = launch_param.stop_iter;
    default_ss.stop_rel_res = launch_param.stop_rel_res_norm;
    default_ss.krylov_dim = launch_param.krylov_dim;
    default_ss.storage_prec = gko::solver::cb_gmres::storage_precision::keep;
    if (launch_param.jacobi_bs <= 0) {
        default_ss.precond = nullptr;
    } else {
        default_ss.precond = precond_type::build()
                                 .with_max_block_size(launch_param.jacobi_bs)
                                 .with_skip_sorting(true)
                                 .on(exec)
                                 ->generate(mtx);
    }
    default_ss.init_compressor = nullptr;

    const auto tt_str = [](int reduction) {
        const std::array<char, 4> types{'d', 'f', 'h', '?'};
        const int base = std::is_same<ValueType, double>::value      ? 0
                         : std::is_same<ValueType, float>::value     ? 1
                         : std::is_same<ValueType, gko::half>::value ? 2
                                                                     : 3;
        if (base == 3) {
            return types[base];
        }
        const int idx = base + reduction;
        return types[idx < types.size() - 1 ? idx : types.size() - 2];
    };
    const std::string str_pre = std::string{"CbGmres<"} + tt_str(0) + ",";
    const std::string str_post{">"};
    const auto get_name = [&str_pre, &str_post, &tt_str](int reduction) {
        return str_pre + tt_str(reduction) + str_post;
    };

    std::cout << data_begin;
    // Make sure the output is in scientific notation for easier comparison
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Matrix" << kv_delim
              << launch_param.matrix_path.substr(
                     launch_param.matrix_path.find_last_of('/') + 1)
              << delim << " size" << kv_delim << mtx_size[0] << " x "
              << mtx_size[1] << delim;
    std::cout << " rhs_norm" << kv_delim << rhs_norm << delim;
    std::cout << " Stopping criteria (iters)" << kv_delim
              << default_ss.stop_iter << delim << " res_norm" << kv_delim
              << default_ss.stop_rel_res << delim;
    std::cout << " Jacobi BS" << kv_delim
              << (default_ss.precond == nullptr
                      ? 0
                      : dynamic_cast<const precond_type*>(
                            default_ss.precond.get())
                            ->get_storage_scheme()
                            .block_offset)
              << " Krylov dim" << kv_delim << default_ss.krylov_dim << '\n';
    const std::array<int, 7> widths{28, 11, 12, 11, 17, 16, 15};
    // Print the header
    // clang-format off
    {
        int i = 0;
        std::cout << std::setw(widths[i++]) << "Name" << delim
                  << std::setw(widths[i++]) << "Comp. info" << delim
                  << std::setw(widths[i++]) << "Time [s]" << delim
                  << std::setw(widths[i++]) << "Iterations" << delim
                  << std::setw(widths[i++]) << "res norm before" << delim
                  << std::setw(widths[i++]) << "res norm after" << delim
                  << std::setw(widths[i++]) << "rel res norm" << delim
                  << " residual norm history (delim: " << res_norm_history_delim
                  << ")" << '\n';
    }
    // clang-format on
    const auto print_result = [&widths](
                                  const std::string& bench_name,
                                  const std::string& comp_info,
                                  const solver_result<RealValueType>& result) {
        int i = 0;
        std::cout << std::setw(widths[i++]) << bench_name << delim
                  << std::setw(widths[i++]) << comp_info << delim
                  << std::setw(widths[i++]) << result.time_s << delim
                  << std::setw(widths[i++]) << result.iters << delim
                  << std::setw(widths[i++]) << result.init_res_norm << delim
                  << std::setw(widths[i++]) << result.res_norm << delim
                  << std::setw(widths[i++])
                  << result.res_norm / result.init_res_norm << delim;
        for (std::size_t i = 0; i < result.residual_norm_history->size(); ++i) {
            if (i != 0) {
                std::cout << res_norm_history_delim;
            }
            std::cout << result.residual_norm_history->at(i);
        }
        std::cout << '\n' << std::flush;
    };

    // Parse File name
    const std::string& config_file = launch_param.compression_json_file;
    auto begin_file_name = config_file.rfind('/');
    begin_file_name =
        begin_file_name == std::string::npos ? 0 : begin_file_name + 1;
    const auto full_file_name = config_file.substr(begin_file_name);
    auto end_file_name = full_file_name.rfind('.');
    end_file_name = end_file_name == std::string::npos ? full_file_name.size()
                                                       : end_file_name;
    const auto file_name = full_file_name.substr(0, end_file_name);
    const std::string file_extension = full_file_name.substr(end_file_name);

    // if the file is named `ieee`, do the IEEE benchmark
    if (full_file_name == "ieee") {
        auto cur_settings = default_ss;
        cur_settings.storage_prec =
            gko::solver::cb_gmres::storage_precision::keep;
        print_result(get_name(0), std::to_string(sizeof(ValueType) * 8),
                     b_object.benchmark_solver(cur_settings));

        if (get_name(0) != get_name(1)) {
            cur_settings.storage_prec =
                gko::solver::cb_gmres::storage_precision::reduce1;
            print_result(get_name(1), std::to_string(sizeof(ValueType) * 8 / 2),
                         b_object.benchmark_solver(cur_settings));
        }

        if (get_name(1) != get_name(2)) {
            cur_settings.storage_prec =
                gko::solver::cb_gmres::storage_precision::reduce2;
            print_result(get_name(2), std::to_string(sizeof(ValueType) * 8 / 4),
                         b_object.benchmark_solver(cur_settings));
        }
    } else if (file_extension != ".json") {
        std::cerr << launch_param.compression_json_file
                  << " is not a JSON file! It must have the '.json' ending!\n";
    } else {
        // search for `__` and separate comp. info from compressor name
        const auto separator_pos = file_name.rfind(file_name_info_separator);
        const auto compression_name = file_name.substr(0, separator_pos);
        const auto comp_info =
            separator_pos == std::string::npos
                ? std::string("file")
                : file_name.substr(separator_pos +
                                   file_name_info_separator.size());
        auto cur_settings = default_ss;
        cur_settings.storage_prec =
            gko::solver::cb_gmres::storage_precision::use_pressio;
        cur_settings.init_compressor = [lp_config =
                                            config_file](void* p_compressor) {
            auto& pc_ = *static_cast<pressio_compressor*>(p_compressor);
            std::ifstream pressio_input_file(lp_config);
            nlohmann::json j;
            pressio_input_file >> j;
            pressio_options options_from_file(static_cast<pressio_options>(j));
            pressio library;
            pc_ = library.get_compressor("pressio");
            // pc_->set_options({});
            pc_->set_name("pressio");
            pc_->set_options(options_from_file);
        };
        print_result(compression_name, comp_info,
                     b_object.benchmark_solver(cur_settings));
    }
    std::cout << data_end;
}


int main(int argc, char* argv[])
{
    user_launch_parameter launch_param{};

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [path/to/matrix.mtx] [path/to/compresson/file.json]"
                     " [stop_iter] [stop_rel_res_norm] [jacobi_block_size] "
                     "[precision={double,float}] "
                     "[exec={cuda,omp,hip,dpcpp,reference}]\n";
        std::cerr << "Default values:"
                  << "\npath/to/matrix.mtx: " << launch_param.matrix_path
                  << "\npath/to/comp/file.json: "
                  << launch_param.compression_json_file
                  << "\nstop_iter: " << launch_param.stop_iter
                  << "\nstop_rel_res_norm: " << launch_param.stop_rel_res_norm
                  << "\njacobi_block_size: " << launch_param.jacobi_bs
                  << "\nprecision: "
                  << "double"
                  << "\nexec: "
                  << "reference" << std::endl;
        std::exit(-1);
    }


    int c_param = 0;  // stores the current parameter index
    if (argc > ++c_param) launch_param.matrix_path = argv[c_param];
    if (argc > ++c_param) launch_param.compression_json_file = argv[c_param];
    if (argc > ++c_param) launch_param.stop_iter = std::stoi(argv[c_param]);
    if (argc > ++c_param)
        launch_param.stop_rel_res_norm = std::stof(argv[c_param]);
    if (argc > ++c_param) launch_param.jacobi_bs = std::stoi(argv[c_param]);

    const std::string precision = argc > ++c_param ? argv[c_param] : "double";
    if (argc > ++c_param) launch_param.exec_string = argv[c_param];

    if (precision == std::string("double")) {
        run_benchmarks<double, int>(launch_param);
    } else if (precision == std::string("float")) {
        run_benchmarks<float, int>(launch_param);
    } else {
        std::cerr << "Unknown precision string \"" << argv[4]
                  << "\". Supported values: \"double\", \"float\"\n";
        std::exit(-1);
    }
}
