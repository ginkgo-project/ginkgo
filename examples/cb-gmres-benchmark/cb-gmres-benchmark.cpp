// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <cinttypes>
#include <climits>  // For CHAR_BIT
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


#include <libpressio_ext/cpp/json.h>
#include <libpressio_ext/cpp/libpressio.h>
#include <libpressio_meta.h>
#include <nlohmann/json.hpp>


template <typename T>
struct float_information {
    static constexpr std::size_t bits = sizeof(T) * CHAR_BIT;
    static std::string get_descr()
    {
        return std::string("float") + std::to_string(bits);
    }
};

template <>
struct float_information<gko::half> {
    static constexpr std::size_t bits = 2 * CHAR_BIT;
    static std::string get_descr()
    {
        return std::string("float") + std::to_string(bits);
    }
};

template <typename T>
struct reduce_float {
    using type = T;
};

template <>
struct reduce_float<double> {
    using type = float;
};

template <>
struct reduce_float<float> {
    using type = gko::half;
};

template <typename T, int Count>
struct reduce_float_by {
    using type = typename reduce_float_by<typename reduce_float<T>::type,
                                          Count - 1>::type;
};

template <typename T>
struct reduce_float_by<T, 0> {
    using type = T;
};

template <typename T, int Count>
using reduce_float_by_t = typename reduce_float_by<T, Count>::type;


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
    double bit_rate;
    T init_res_norm;
    T res_norm;
    std::vector<T> residual_norm_history;
    std::map<std::string, double> operation_timings;
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

    std::vector<RealValueType> get_residual_norm_history_copy() const
    {
        return residual_norm_history_;
    }

    const std::vector<RealValueType>& get_residual_norm_history()
    {
        return residual_norm_history_;
    }

    std::vector<RealValueType> extract_residual_norm_history()
    {
        return std::move(residual_norm_history_);
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

// A logger that accumulates the time of all operations. For each operation type
// (allocations, free, copy, internal operations i.e. kernels), the timing is
// taken before and after. This can create significant overhead since to ensure
// proper timings, calls to `synchronize` are required.
struct OperationLogger : gko::log::Logger {
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type&) const override
    {
        this->start_operation(exec, "allocate");
    }

    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        this->end_operation(exec, "allocate");
    }

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr&) const override
    {
        this->start_operation(exec, "free");
    }

    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr&) const override
    {
        this->end_operation(exec, "free");
    }

    void on_copy_started(const gko::Executor* from, const gko::Executor* to,
                         const gko::uintptr&, const gko::uintptr&,
                         const gko::size_type&) const override
    {
        from->synchronize();
        this->start_operation(to, "copy");
    }

    void on_copy_completed(const gko::Executor* from, const gko::Executor* to,
                           const gko::uintptr&, const gko::uintptr&,
                           const gko::size_type&) const override
    {
        from->synchronize();
        this->end_operation(to, "copy");
    }

    void on_operation_launched(const gko::Executor* exec,
                               const gko::Operation* op) const override
    {
        this->start_operation(exec, op->get_name());
    }

    void on_operation_completed(const gko::Executor* exec,
                                const gko::Operation* op) const override
    {
        this->end_operation(exec, op->get_name());
    }

    void write_data(std::ostream& ostream)
    {
        for (const auto& entry : total) {
            ostream << "\t" << entry.first.c_str() << ": "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           entry.second)
                           .count()
                    << std::endl;
        }
    }

    std::map<std::string, double> get_duration_map_s() const
    {
        std::map<std::string, double> result;
        for (auto&& entry : total) {
            result.insert({entry.first, entry.second.count()});
        }
        return result;
    }

    void reset()
    {
        start.clear();
        total.clear();
        nested.clear();
    }

private:
    // Helper which synchronizes and starts the time before every operation.
    void start_operation(const gko::Executor* exec,
                         const std::string& name) const
    {
        nested.emplace_back(0);
        exec->synchronize();
        start[name] = std::chrono::steady_clock::now();
    }

    // Helper to compute the end time and store the operation's time at its
    // end. Also time nested operations.
    void end_operation(const gko::Executor* exec, const std::string& name) const
    {
        exec->synchronize();
        const auto end = std::chrono::steady_clock::now();
        const auto diff = end - start[name];
        // make sure timings for nested operations are not counted twice
        total[name] += diff - nested.back();
        nested.pop_back();
        if (nested.size() > 0) {
            nested.back() += diff;
        }
    }

    mutable std::map<std::string, std::chrono::steady_clock::time_point> start;
    mutable std::map<std::string, std::chrono::duration<double>> total;
    // the position i of this vector holds the total time spend on child
    // operations on nesting level i
    mutable std::vector<std::chrono::duration<double>> nested;
};


template <typename ValueType, typename IndexType = int>
class Benchmark {
private:
    using RealValueType = gko::remove_complex<ValueType>;
    using Vector = gko::matrix::Dense<ValueType>;
    using NormVector = gko::matrix::Dense<RealValueType>;
    static constexpr int repeats{1};

public:
    Benchmark(std::shared_ptr<gko::Executor> exec,
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
              ConvergenceHistoryLogger<ValueType>::create()},
          operation_logger_{nullptr}
    {
        operation_logger_ = std::make_shared<OperationLogger>();
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

        // Set operation_logger_ to nullptr to disable logging
        if (operation_logger_) {
            exec_->add_logger(operation_logger_);
        }


        for (int i = 0; i < repeats; ++i) {
            // No need to copy it in the first iteration
            if (i != 0) {
                x_->copy_from(init_x_.get());
                convergence_history_logger_.reset();
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
        iter_stop->remove_logger(convergence_history_logger_);

        result.iters = convergence_history_logger_->get_num_iterations();
        result.time_s = duration / static_cast<double>(repeats);
        result.bit_rate = solver->get_average_bit_rate();
        result.res_norm = this->compute_residual_norm();
        result.residual_norm_history =
            convergence_history_logger_->extract_residual_norm_history();
        if (operation_logger_) {
            result.operation_timings = operation_logger_->get_duration_map_s();
            exec_->remove_logger(operation_logger_);
        }
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
        if (operation_logger_) {
            operation_logger_->reset();
        }
    }

    std::shared_ptr<gko::Executor> exec_;
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
    std::shared_ptr<OperationLogger> operation_logger_;
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
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    auto exec = exec_map.at(launch_param.exec_string)();  // throws if not valid

    // constexpr char delim = ';';
    // constexpr char res_norm_history_delim = '&';
    // constexpr char kv_delim = ':';
    // const std::string data_begin = "{\n";
    // const std::string data_end = "}\n";
    const std::string file_name_info_separator = "__";

    auto mtx =
        share(gko::read<mtx_t>(std::ifstream(launch_param.matrix_path), exec));

    const auto mtx_size = mtx->get_size();
    auto rhs = vec::create(exec, gko::dim<2>{mtx_size[0], 1});
    auto init_x = vec::create(exec, gko::dim<2>{mtx_size[1], 1});


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
        mtx->apply(res_host, rhs);

        // As an initial guess, use the right-hand side
        auto init_x_host = clone(exec->get_master(), init_x);
        for (gko::size_type i = 0; i < init_x_host->get_size()[0]; ++i) {
            init_x_host->at(i, 0) = 0;
        }
        init_x->copy_from(init_x_host);
    }

    Benchmark b_object(exec, mtx, std::move(init_x), std::move(rhs));

    std::string matrix_name = launch_param.matrix_path.substr(
        launch_param.matrix_path.find_last_of('/') + 1);
    if (matrix_name.substr(matrix_name.size() - 4) == ".mtx") {
        matrix_name = matrix_name.substr(0, matrix_name.size() - 4);
    }
    using precond_type = gko::preconditioner::Jacobi<ValueType, IndexType>;
    // Default_settings
    solver_settings default_ss{};
    default_ss.stop_iter = launch_param.stop_iter;
    default_ss.stop_rel_res = launch_param.stop_rel_res_norm;
    default_ss.krylov_dim = launch_param.krylov_dim;
    default_ss.storage_prec = gko::solver::cb_gmres::storage_precision::keep;
    default_ss.precond = nullptr;
    default_ss.init_compressor = nullptr;

    nlohmann::json global_settings = {
        {"matrix", matrix_name},
        {"size", {{"rows", mtx_size[0]}, {"cols", mtx_size[1]}}},
        {"rhs_norm", b_object.get_rhs_norm()},
        {"stopping_iters", default_ss.stop_iter},
        {"stopping_res_norm", default_ss.stop_rel_res},
        {"stopping_res_norm_baseline", "rhs_norm"},
        {"krylov_dim", default_ss.krylov_dim},
        {"arithmetic_type", float_information<ValueType>::get_descr()},
        {"precond",
         {{"name", "unknown"}, {"settings", nlohmann::json::object()}}},
    };

    if (launch_param.jacobi_bs <= 0) {
        default_ss.precond = nullptr;
        global_settings["precond"]["name"] = "none";
    } else {
        default_ss.precond = precond_type::build()
                                 .with_max_block_size(launch_param.jacobi_bs)
                                 .with_skip_sorting(true)
                                 .on(exec)
                                 ->generate(mtx);
        global_settings["precond"]["name"] = "block-jacobi";
        global_settings["precond"]["settings"]["block_size"] =
            std::dynamic_pointer_cast<const precond_type>(default_ss.precond)
                ->get_parameters()
                .max_block_size;
    }


    const auto get_result_json =
        [rhs_norm = b_object.get_rhs_norm()](
            const std::string& bench_name,
            solver_result<RealValueType>&& result) -> nlohmann::json {
        nlohmann::json json_result = {
            {"name", bench_name},
            {"settings", nlohmann::json::object()},
            {"bit_rate", result.bit_rate},
            {"time_s", result.time_s},
            {"iterations", result.iters},
            {"init_res_norm", result.init_res_norm},
            {"final_res_norm", result.res_norm},
            {"rel_res_norm", result.res_norm / rhs_norm},
            {"res_norm_history", std::move(result.residual_norm_history)},
        };
        if (result.operation_timings.size() > 0) {
            json_result["operations"] = result.operation_timings;
        }
        //  The actual settings information (the JSON file used for the
        //  compression) needs to be added afterwards
        return json_result;
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

    nlohmann::json results;
    // if the file is named `ieee`, do the IEEE benchmark
    if (full_file_name == "ieee") {
        auto cur_settings = default_ss;
        cur_settings.storage_prec =
            gko::solver::cb_gmres::storage_precision::keep;
        using f_info = float_information<reduce_float_by_t<ValueType, 0>>;
        results.push_back(get_result_json(
            f_info::get_descr(), b_object.benchmark_solver(cur_settings)));

        if (!std::is_same<ValueType, reduce_float_by_t<ValueType, 1>>::value) {
            cur_settings.storage_prec =
                gko::solver::cb_gmres::storage_precision::reduce1;
            using f_info = float_information<reduce_float_by_t<ValueType, 1>>;
            results.push_back(get_result_json(
                f_info::get_descr(), b_object.benchmark_solver(cur_settings)));
        }

        if (!std::is_same<ValueType, reduce_float_by_t<ValueType, 2>>::value) {
            cur_settings.storage_prec =
                gko::solver::cb_gmres::storage_precision::reduce2;
            using f_info = float_information<reduce_float_by_t<ValueType, 2>>;
            results.push_back(get_result_json(
                f_info::get_descr(), b_object.benchmark_solver(cur_settings)));
        }
    } else if (full_file_name == "frsz2_21" || full_file_name == "frsz2_32") {
        auto cur_settings = default_ss;
        cur_settings.storage_prec =
            full_file_name == "frsz2_21"
                ? gko::solver::cb_gmres::storage_precision::use_frsz2_21
                : gko::solver::cb_gmres::storage_precision::use_frsz2_32;
        using f_info = float_information<reduce_float_by_t<ValueType, 0>>;
        results.push_back(get_result_json(
            full_file_name, b_object.benchmark_solver(cur_settings)));
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
        std::ifstream pressio_input_file(config_file);
        nlohmann::json settings_json;
        pressio_input_file >> settings_json;
        cur_settings.init_compressor = [settings_json](void* p_compressor) {
            auto& pc_ = *static_cast<pressio_compressor*>(p_compressor);
            pressio_options options_from_file(
                static_cast<pressio_options>(settings_json));
            pressio library;
            pc_ = library.get_compressor("pressio");
            // pc_->set_options({});
            // pc_->set_name("pressio");
            pc_->set_options(options_from_file);
            // std::cout << pc_->get_options() << '\n';
        };
        auto current_result = get_result_json(
            compression_name, b_object.benchmark_solver(cur_settings));
        current_result["settings"] = settings_json;
        results.push_back(current_result);
    }
    nlohmann::json global_output = {{"settings", global_settings},
                                    {"results", results}};
    std::cout << global_output.dump();
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
