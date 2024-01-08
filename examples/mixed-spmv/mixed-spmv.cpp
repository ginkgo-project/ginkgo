// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection.
#include <map>
// Add the string manipulation header to handle strings.
#include <string>
// Add the timing header for timing.
#include <chrono>
// Add the random header to generate random vectors.
#include <random>

namespace {


/**
 * Generate a random value.
 *
 * @tparam ValueType  valuetype of the value
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param value_dist  distribution of array values
 * @param engine  a random engine
 *
 * @return ValueType
 */
template <typename ValueType, typename ValueDistribution, typename Engine>
typename std::enable_if<!gko::is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(ValueDistribution&& value_dist, Engine&& gen)
{
    return value_dist(gen);
}

/**
 * Specialization for complex types.
 *
 * @copydoc get_rand_value
 */
template <typename ValueType, typename ValueDistribution, typename Engine>
typename std::enable_if<gko::is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(ValueDistribution&& value_dist, Engine&& gen)
{
    return ValueType(value_dist(gen), value_dist(gen));
}

/**
 * timing the apply operation A->apply(b, x). It will runs 2 warmup and get
 * average time among 10 times.
 *
 * @return seconds
 */
double timing(std::shared_ptr<const gko::Executor> exec,
              std::shared_ptr<const gko::LinOp> A,
              std::shared_ptr<const gko::LinOp> b,
              std::shared_ptr<gko::LinOp> x)
{
    int warmup = 2;
    int rep = 10;
    for (int i = 0; i < warmup; i++) {
        A->apply(b, x);
    }
    double total_sec = 0;
    for (int i = 0; i < rep; i++) {
        // always clone the x in each apply
        auto xx = x->clone();
        // synchronize to make sure data is already on device
        exec->synchronize();
        auto start = std::chrono::steady_clock::now();
        A->apply(b, xx);
        // synchronize to make sure the operation is done
        exec->synchronize();
        auto stop = std::chrono::steady_clock::now();
        // get the duration in seconds
        std::chrono::duration<double> duration_time = stop - start;
        total_sec += duration_time.count();
        if (i + 1 == rep) {
            // copy the result back to x
            x->copy_from(xx);
        }
    }

    return total_sec / rep;
}


}  // namespace


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using HighPrecision = double;
    using RealValueType = gko::remove_complex<HighPrecision>;
    using LowPrecision = float;
    using IndexType = int;
    using hp_vec = gko::matrix::Dense<HighPrecision>;
    using lp_vec = gko::matrix::Dense<LowPrecision>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Ell class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Csr or
    // gko::matrix::Sellp could also be used.
    // Note. the behavior will depends GINKGO_MIXED_PRECISION flags and the
    // actual implementation from different matrices.
    using hp_mtx = gko::matrix::Ell<HighPrecision, IndexType>;
    using lp_mtx = gko::matrix::Ell<LowPrecision, IndexType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your operation?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // @sect3{Preparing your data and transfer to the proper device.}
    // Read the matrix using the @ref read function and set the right hand side
    // randomly.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. gko::share and gko::give are the functions that you would need
    // to use.

    // read the matrix into HighPrecision and LowPrecision.
    auto hp_A = share(gko::read<hp_mtx>(std::ifstream("data/A.mtx"), exec));
    auto lp_A = share(gko::read<lp_mtx>(std::ifstream("data/A.mtx"), exec));
    // Set the shortcut for each dimension
    auto A_dim = hp_A->get_size();
    auto b_dim = gko::dim<2>{A_dim[1], 1};
    auto x_dim = gko::dim<2>{A_dim[0], b_dim[1]};
    auto host_b = hp_vec::create(exec->get_master(), b_dim);
    // fill the b vector with some random data
    std::default_random_engine rand_engine(32);
    auto dist = std::uniform_real_distribution<RealValueType>(0.0, 1.0);
    for (int i = 0; i < host_b->get_size()[0]; i++) {
        host_b->at(i, 0) = get_rand_value<HighPrecision>(dist, rand_engine);
    }
    // copy the data from host to device
    auto hp_b = share(gko::clone(exec, host_b));
    auto lp_b = share(lp_vec::create(exec));
    lp_b->copy_from(hp_b);

    // create several result x vector in different precision
    auto hp_x = share(hp_vec::create(exec, x_dim));
    auto lp_x = share(lp_vec::create(exec, x_dim));
    auto hplp_x = share(hp_x->clone());
    auto lplp_x = share(hp_x->clone());
    auto lphp_x = share(hp_x->clone());

    // @sect3{Measure the time of apply}
    // We measure the time among different combination of apply operation.

    // Hp * Hp -> Hp
    auto hp_sec = timing(exec, hp_A, hp_b, hp_x);
    // Lp * Lp -> Lp
    auto lp_sec = timing(exec, lp_A, lp_b, lp_x);
    // Hp * Lp -> Hp
    auto hplp_sec = timing(exec, hp_A, lp_b, hplp_x);
    // Lp * Lp -> Hp
    auto lplp_sec = timing(exec, lp_A, lp_b, lplp_x);
    // Lp * Hp -> Hp
    auto lphp_sec = timing(exec, lp_A, hp_b, lphp_x);


    // To measure error of result.
    // neg_one is an object that represent the number -1.0 which allows for a
    // uniform interface when computing on any device. To compute the residual,
    // all you need to do is call the add_scaled method, which in this case is
    // an axpy and equivalent to the LAPACK axpy routine. Finally, you compute
    // the euclidean 2-norm with the compute_norm2 function.
    auto neg_one = gko::initialize<hp_vec>({-1.0}, exec);
    auto hp_x_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
    auto lp_diff_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
    auto hplp_diff_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
    auto lplp_diff_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
    auto lphp_diff_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
    auto lp_diff = hp_x->clone();
    auto hplp_diff = hp_x->clone();
    auto lplp_diff = hp_x->clone();
    auto lphp_diff = hp_x->clone();

    hp_x->compute_norm2(hp_x_norm);
    lp_diff->add_scaled(neg_one, lp_x);
    lp_diff->compute_norm2(lp_diff_norm);
    hplp_diff->add_scaled(neg_one, hplp_x);
    hplp_diff->compute_norm2(hplp_diff_norm);
    lplp_diff->add_scaled(neg_one, lplp_x);
    lplp_diff->compute_norm2(lplp_diff_norm);
    lphp_diff->add_scaled(neg_one, lphp_x);
    lphp_diff->compute_norm2(lphp_diff_norm);
    exec->synchronize();

    std::cout.precision(10);
    std::cout << std::scientific;
    std::cout << "High Precision time(s): " << hp_sec << std::endl;
    std::cout << "High Precision result norm: " << hp_x_norm->at(0)
              << std::endl;
    std::cout << "Low Precision time(s): " << lp_sec << std::endl;
    std::cout << "Low Precision relative error: "
              << lp_diff_norm->at(0) / hp_x_norm->at(0) << "\n";
    std::cout << "Hp * Lp -> Hp time(s): " << hplp_sec << std::endl;
    std::cout << "Hp * Lp -> Hp relative error: "
              << hplp_diff_norm->at(0) / hp_x_norm->at(0) << "\n";
    std::cout << "Lp * Lp -> Hp time(s): " << lplp_sec << std::endl;
    std::cout << "Lp * Lp -> Hp relative error: "
              << lplp_diff_norm->at(0) / hp_x_norm->at(0) << "\n";
    std::cout << "Lp * Hp -> Hp time(s): " << lplp_sec << std::endl;
    std::cout << "Lp * Hp -> Hp relative error: "
              << lphp_diff_norm->at(0) / hp_x_norm->at(0) << "\n";
}
