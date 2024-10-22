// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <variant>

#include <cxxopts.hpp>

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include "batched/batch_cg.hpp"


namespace dummy {
struct custom_operator_view {
    gko::size_type num_batch_items;
    gko::int32 num_rows;
    gko::int32 num_cols;
};

class CustomOperator : public gko::EnablePolymorphicObject<CustomOperator> {
public:
    using value_type = double;

    struct const_item {};

    explicit CustomOperator(std::shared_ptr<const gko::Executor> exec,
                            gko::batch_dim<2> size = {})
        : EnablePolymorphicObject(std::move(exec)), size_(size)
    {}

    [[nodiscard]] constexpr custom_operator_view create_view() const
    {
        return {this->get_num_batch_items(),
                static_cast<gko::int32>(this->get_common_size()[0]),
                static_cast<gko::int32>(this->get_common_size()[1])};
    }

    [[nodiscard]] constexpr gko::batch_dim<2> get_size() const { return size_; }

    [[nodiscard]] constexpr gko::dim<2> get_common_size() const
    {
        return size_.get_common_size();
    }

    [[nodiscard]] constexpr gko::size_type get_num_batch_items() const
    {
        return size_.get_num_batch_items();
    }

private:
    gko::batch_dim<2> size_;
};

struct custom_operator_item {
    gko::size_type num_batches;
    gko::size_type batch_id;
    gko::int32 num_rows;
    gko::int32 num_cols;
};

constexpr custom_operator_item extract_batch_item(custom_operator_view op,
                                                  gko::size_type batch_id)
{
    return {op.num_batch_items, batch_id, op.num_rows, op.num_cols};
}


constexpr void advanced_apply(
    double alpha, custom_operator_item a,
    gko::batch::multi_vector::batch_item<const double> b, double beta,
    gko::batch::multi_vector::batch_item<double> x,
    [[maybe_unused]] std::variant<gko::reference_kernel, gko::omp_kernel>)
{
    auto num_batches = a.num_batches;
    for (gko::size_type row = 0; row < a.num_rows; ++row) {
        double acc{};

        if (row > 0) {
            acc += -gko::one<double>() * b.values[row - 1];
        }
        acc +=
            (static_cast<double>(2.0) + static_cast<double>(a.batch_id) /
                                            static_cast<double>(num_batches)) *
            b.values[row];
        if (row < a.num_rows - 1) {
            acc += -gko::one<double>() * b.values[row + 1];
        }
        x.values[row] = alpha * acc + beta * x.values[row];
    }
}

constexpr void simple_apply(
    const custom_operator_item& a,
    const gko::batch::multi_vector::batch_item<const double>& b,
    const gko::batch::multi_vector::batch_item<double>& x,
    [[maybe_unused]] std::variant<gko::reference_kernel, gko::omp_kernel>)
{
    advanced_apply(1.0, a, b, 0.0, x, gko::reference_kernel{});
}


__device__ void advanced_apply(
    double alpha, custom_operator_item a,
    gko::batch::multi_vector::batch_item<const double> b, double beta,
    gko::batch::multi_vector::batch_item<double> x,
    [[maybe_unused]] gko::cuda_hip_kernel)
{
    auto tidx = threadIdx.x;
    auto num_batches = a.num_batches;
    for (gko::size_type row = tidx; row < a.num_rows; row += blockDim.x) {
        double acc{};

        if (row > 0) {
            acc += -gko::one<double>() * b.values[row - 1];
        }
        acc +=
            (static_cast<double>(2.0) + static_cast<double>(a.batch_id) /
                                            static_cast<double>(num_batches)) *
            b.values[row];
        if (row < a.num_rows - 1) {
            acc += -gko::one<double>() * b.values[row + 1];
        }
        // auto dummy = alpha * acc + beta * x[row];
        x.values[row] = alpha * acc + beta * x.values[row];
    }
}

__device__ void simple_apply(
    const custom_operator_item& a,
    const gko::batch::multi_vector::batch_item<const double>& b,
    const gko::batch::multi_vector::batch_item<double>& x,
    [[maybe_unused]] gko::cuda_hip_kernel tag)
{
    advanced_apply(1.0, a, b, 0.0, x, tag);
}


}  // namespace dummy


// @sect3{Type aliases for convenience}
// Use some shortcuts.
using value_type = double;
using real_type = gko::remove_complex<value_type>;
using index_type = int;
using size_type = gko::size_type;
using vec_type = gko::batch::MultiVector<value_type>;
using real_vec_type = gko::batch::MultiVector<real_type>;
using mtx_type = gko::batch::matrix::Csr<value_type, index_type>;
using op_type = dummy::CustomOperator;
using cg = gko::batch_template::solver::Cg<mtx_type>;
using cg_op = gko::batch_template::solver::Cg<op_type>;


// @sect3{Convenience functions}
// Unbatch batch items into distinct Ginkgo types
namespace detail {


template <typename InputType>
auto unbatch(const InputType* batch_object)
{
    auto unbatched_mats =
        std::vector<std::unique_ptr<typename InputType::unbatch_type>>{};
    for (size_type b = 0; b < batch_object->get_num_batch_items(); ++b) {
        unbatched_mats.emplace_back(
            batch_object->create_const_view_for_item(b)->clone());
    }
    return unbatched_mats;
}


}  // namespace detail


int main(int argc, char* argv[])
{
    // Print ginkgo version information
    std::cout << gko::version_info::get() << std::endl;

    cxxopts::Options options(
        "batched-matrix-free",
        "Solve a batched problem with the external matrix format.");

    options.add_options()(
        "executor", "The Ginkgo Executor type",
        cxxopts::value<std::string>()->default_value("reference"))(
        "b,batches", "The number of batches",
        cxxopts::value<value_type>()->default_value("2"))(
        "s,size", "The (square) size of a batch item",
        cxxopts::value<size_type>()->default_value("32"))(
        "print-residuals", "Print the final residuals",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "matrix-free", "Use external matrix format",
        cxxopts::value<bool>()->default_value("true")->implicit_value("true"))(
        "h,help", "Show this message");
    options.parse_positional("executor");
    options.allow_unrecognised_options();
    auto args = options.parse(argc, argv);

    if (args.count("help") || !args.unmatched().empty()) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for an gko::OmpExecutor, which uses OpenMP
    // multi-threading in most of its kernels,
    // a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor, gko::CudaExecutor, gko::HipExecutor,
    // gko::DpcppExecutor which runs the code on a NVIDIA, AMD and Intel GPUs,
    // respectively.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = args["executor"].as<std::string>();
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto num_systems = static_cast<size_type>(args["batches"].as<value_type>());
    auto num_rows = args["size"].as<size_type>();

    if (num_rows < 2) {
        std::cerr << "Expected a <num_rows> to be >= 2." << std::endl;
        std::exit(-1);
    }

    // @sect3{Generate data}
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, num_rows));
    auto batch_vec_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, 1));
    auto nnz = 3 * (num_rows - 2) + 4;

    auto A_mtx =
        gko::share(mtx_type::create(exec->get_master(), batch_mat_size, nnz));
    auto A_op = std::make_shared<op_type>(exec, batch_mat_size);

    if (args["print-residuals"].as<bool>() || !args["matrix-free"].as<bool>()) {
        gko::matrix_data<value_type, index_type> md(
            batch_mat_size.get_common_size());
        auto create_batch = [batch_mat_size, num_rows, nnz](gko::size_type id,
                                                            auto& md) {
            md.nonzeros.reserve(nnz);
            for (index_type i = 0; i < num_rows; ++i) {
                if (i > 0) {
                    md.nonzeros.emplace_back(i, i - 1, -1);
                }
                md.nonzeros.emplace_back(
                    i, i,
                    2 + value_type(id) / batch_mat_size.get_num_batch_items());
                if (i < num_rows - 1) {
                    md.nonzeros.emplace_back(i, i + 1, -1);
                }
            }
        };
#pragma omp parallel for firstprivate(md)
        for (gko::size_type id = 0; id < batch_mat_size.get_num_batch_items();
             ++id) {
            md.nonzeros.clear();
            create_batch(id, md);
            A_mtx->create_view_for_item(id)->read(md);
        }
        A_mtx = gko::clone(exec, A_mtx);
    }

    // @sect3{RHS and solution vectors}
    auto b = vec_type::create(exec, batch_vec_size);
    b->fill(1.0);
    // Create initial guess as 0 and copy to device
    auto x = vec_type::create(exec, batch_vec_size);
    x->fill(0.0);

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-10};
    // Create a batched solver factory with relevant parameters.
    using SolverVariant =
        std::variant<std::shared_ptr<cg>, std::shared_ptr<cg_op>>;
    auto solver{
        args["matrix-free"].as<bool>()
            ? SolverVariant(cg_op::build()
                                .with_max_iterations(500)
                                .with_tolerance(reduction_factor)
                                .with_tolerance_type(
                                    gko::batch::stop::tolerance_type::relative)
                                .on(exec)
                                ->generate(A_op))
            : SolverVariant(cg::build()
                                .with_max_iterations(500)
                                .with_tolerance(reduction_factor)
                                .with_tolerance_type(
                                    gko::batch::stop::tolerance_type::relative)
                                .on(exec)
                                ->generate(A_mtx))};

    // @sect3{Batch logger}
    // Create a logger to obtain the iteration counts and "implicit" residual
    //  norms for every system after the solve.
    std::shared_ptr<const gko::batch::log::BatchConvergence<value_type>>
        logger = gko::batch::log::BatchConvergence<value_type>::create();

    // @sect3{Generate and solve}
    // add the logger to the solver
    std::visit([logger](auto& solver_v) { solver_v->add_logger(logger); },
               solver);


    exec->synchronize();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    std::visit([&b, &x](auto& solver_v) { solver_v->apply(b, x); }, solver);

    exec->synchronize();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "Solver type: "
              << "batch::cg"
              << "\nMatrix size: " << A_mtx->get_common_size()
              << "\nNum batch entries: " << A_mtx->get_num_batch_items()
              << "\nTime elapsed: " << time_span.count() << std::endl;

    if (args["print-residuals"].as<bool>()) {
        // @sect3{Check result}
        // Compute norm of RHS on the device and automatically copy to host
        auto norm_dim = gko::batch_dim<2>(num_systems, gko::dim<2>(1, 1));
        auto host_b_norm = real_vec_type::create(exec->get_master(), norm_dim);
        host_b_norm->fill(0.0);

        b->compute_norm2(host_b_norm);
        // we need constants on the device
        auto one = vec_type::create(exec, norm_dim);
        one->fill(1.0);
        auto neg_one = vec_type::create(exec, norm_dim);
        neg_one->fill(-1.0);
        // allocate and compute the residual
        auto res = vec_type::create(exec, batch_vec_size);
        res->copy_from(b);
        // @todo: change this when the external apply becomes available
        A_mtx->apply(one, x, neg_one, res);
        // allocate and compute residual norm
        auto host_res_norm =
            real_vec_type::create(exec->get_master(), norm_dim);
        host_res_norm->fill(0.0);
        res->compute_norm2(host_res_norm);
        auto host_log_resid = gko::make_temporary_clone(
            exec->get_master(), &logger->get_residual_norm());
        auto host_log_iters = gko::make_temporary_clone(
            exec->get_master(), &logger->get_num_iterations());

        std::cout << "Residual norm sqrt(r^T r):\n";
        // "unbatch" converts a batch object into a vector of objects of the
        // corresponding single type, eg. batch::matrix::Dense -->
        // std::vector<Dense>.
        auto unb_res = detail::unbatch(host_res_norm.get());
        auto unb_bnorm = detail::unbatch(host_b_norm.get());
        for (size_type i = 0; i < num_systems; ++i) {
            std::cout << " System no. " << i
                      << ": residual norm = " << unb_res[i]->at(0, 0)
                      << ", implicit residual norm = "
                      << host_log_resid->get_const_data()[i]
                      << ", iterations = "
                      << host_log_iters->get_const_data()[i] << std::endl;
            const real_type relresnorm =
                unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
            if (!(relresnorm <= reduction_factor)) {
                std::cout << "System " << i << " converged only to "
                          << relresnorm << " relative residual." << std::endl;
            }
        }
    }

    return 0;
}
