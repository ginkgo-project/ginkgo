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

#include <iostream>
#include <map>
#include <string>


#include <omp.h>
#include <ginkgo/ginkgo.hpp>


// A CUDA kernel implementing the stencil, which will be used if running on the
// CUDA executor. Unfortunately, NVCC has serious problems interpreting some
// parts of Ginkgo's code, so the kernel has to be compiled separately.
template <typename ValueType>
std::shared_ptr<gko::AsyncHandle> stencil_kernel(
    std::size_t size, const ValueType* coefs, const ValueType* b, ValueType* x,
    std::shared_ptr<gko::AsyncHandle> handle);


// A stencil matrix class representing the 3pt stencil linear operator.
// We include the gko::EnableLinOp mixin which implements the entire LinOp
// interface, except the two apply_impl methods, which get called inside the
// default implementation of apply (after argument verification) to perform the
// actual application of the linear operator. In addition, it includes the
// implementation of the entire PolymorphicObject interface.
//
// It also includes the gko::EnableCreateMethod mixin which provides a default
// implementation of the static create method. This method will forward all its
// arguments to the constructor to create the object, and return an
// std::unique_ptr to the created object.
template <typename ValueType>
class StencilMatrix : public gko::EnableLinOp<StencilMatrix<ValueType>>,
                      public gko::EnableCreateMethod<StencilMatrix<ValueType>> {
public:
    // This constructor will be called by the create method. Here we initialize
    // the coefficients of the stencil.
    StencilMatrix(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type size = 0, ValueType left = -1.0,
                  ValueType center = 2.0, ValueType right = -1.0)
        : gko::EnableLinOp<StencilMatrix>(exec, gko::dim<2>{size}),
          coefficients(exec, {left, center, right})
    {}

protected:
    using vec = gko::matrix::Dense<ValueType>;
    using coef_type = gko::Array<ValueType>;

    struct stencil_operation : gko::AsyncOperation {
        stencil_operation(const coef_type& coefficients, const vec* b, vec* x)
            : coefficients{coefficients}, b{b}, x{x}
        {}

        // OpenMP implementation
        std::shared_ptr<gko::AsyncHandle> run(
            std::shared_ptr<const gko::OmpExecutor>,
            std::shared_ptr<gko::AsyncHandle> handle) const override
        {
            auto l = [=]() {
                auto b_values = b->get_const_values();
                auto x_values = x->get_values();
#pragma omp parallel for
                for (std::size_t i = 0; i < x->get_size()[0]; ++i) {
                    auto coefs = coefficients.get_const_data();
                    auto result = coefs[1] * b_values[i];
                    if (i > 0) {
                        result += coefs[0] * b_values[i - 1];
                    }
                    if (i < x->get_size()[0] - 1) {
                        result += coefs[2] * b_values[i + 1];
                    }
                    x_values[i] = result;
                }
            };

            return gko::as<gko::HostAsyncHandle<void>>(handle)->queue(
                std::async(std::launch::async, l));
        }

        // CUDA implementation
        std::shared_ptr<gko::AsyncHandle> run(
            std::shared_ptr<const gko::CudaExecutor> exec,
            std::shared_ptr<gko::AsyncHandle> handle) const override
        {
            return stencil_kernel(
                x->get_size()[0], coefficients.get_const_data(),
                b->get_const_values(), x->get_values(), handle);
        }

        // We do not provide an implementation for reference executor.
        // If not provided, Ginkgo will use the implementation for the
        // OpenMP executor when calling it in the reference executor.

        const coef_type& coefficients;
        const vec* b;
        vec* x;
    };

    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        // we only implement the operator for dense RHS.
        // gko::as will throw an exception if its argument is not Dense.
        auto exec = this->get_executor();
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);

        // we need separate implementations depending on the executor, so we
        // create an operation which maps the call to the correct implementation
        exec->run(stencil_operation(coefficients, dense_b, dense_x),
                  exec->get_default_exec_stream());
        exec->get_default_exec_stream()->wait();
    }

    std::shared_ptr<gko::AsyncHandle> apply_impl(
        const gko::LinOp* b, gko::LinOp* x,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        // we only implement the operator for dense RHS.
        // gko::as will throw an exception if its argument is not Dense.
        auto exec = this->get_executor();
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);

        // we need separate implementations depending on the executor, so we
        // create an operation which maps the call to the correct implementation
        return exec->run(stencil_operation(coefficients, dense_b, dense_x),
                         handle);
    }

    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto tmp_x = dense_x->clone();
        this->apply_impl(b, lend(tmp_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(tmp_x));
    }

private:
    coef_type coefficients;
};


// Creates a stencil matrix in CSR format for the given number of discretization
// points.
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(gko::matrix::Csr<ValueType, IndexType>* matrix)
{
    const auto discretization_points = matrix->get_size()[0];
    auto row_ptrs = matrix->get_row_ptrs();
    auto col_idxs = matrix->get_col_idxs();
    auto values = matrix->get_values();
    IndexType pos = 0;
    const ValueType coefs[] = {-1, 2, -1};
    row_ptrs[0] = pos;
    for (int i = 0; i < discretization_points; ++i) {
        for (auto ofs : {-1, 0, 1}) {
            if (0 <= i + ofs && i + ofs < discretization_points) {
                values[pos] = coefs[ofs + 1];
                col_idxs[pos] = i + ofs;
                ++pos;
            }
        }
        row_ptrs[i + 1] = pos;
    }
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ValueType>
void generate_rhs(Closure f, ValueType u0, ValueType u1,
                  gko::matrix::Dense<ValueType>* rhs)
{
    const auto discretization_points = rhs->get_size()[0];
    auto values = rhs->get_values();
    const ValueType h = 1.0 / (discretization_points + 1);
    for (int i = 0; i < discretization_points; ++i) {
        const ValueType xi = ValueType(i + 1) * h;
        values[i] = -f(xi) * h * h;
    }
    values[0] += u0;
    values[discretization_points - 1] += u1;
}


// Prints the solution `u`.
template <typename ValueType>
void print_solution(ValueType u0, ValueType u1,
                    const gko::matrix::Dense<ValueType>* u)
{
    std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    std::cout << u1 << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType>
double calculate_error(int discretization_points,
                       const gko::matrix::Dense<ValueType>* u,
                       Closure correct_u)
{
    const auto h = 1.0 / (discretization_points + 1);
    auto error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        const auto xi = (i + 1) * h;
        error +=
            abs(u->get_const_values()[i] - correct_u(xi)) / abs(correct_u(xi));
    }
    return error;
}


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const unsigned int discretization_points =
        argc >= 3 ? std::atoi(argv[2]) : 25u;
    const unsigned int num_reps = argc >= 4 ? std::atoi(argv[3]) : 10u;
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(
                     0, gko::OmpExecutor::create(), true,
                     gko::allocation_mode::device, 2);
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
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    // executor used by the application
    const auto app_exec = exec->get_master();

    // problem:
    auto correct_u = [](ValueType x) { return x * x * x; };
    auto f = [](ValueType x) { return ValueType{6} * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // initialize vectors
    auto rhs_h = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    generate_rhs(f, u0, u1, lend(rhs_h));
    auto rhs = vec::create(exec, gko::dim<2>(discretization_points, 1));
    rhs->copy_from(rhs_h.get());
    auto rhs2 = vec::create(exec, gko::dim<2>(discretization_points, 1));
    rhs2->copy_from(rhs_h.get());
    auto uh = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    for (int i = 0; i < uh->get_size()[0]; ++i) {
        uh->get_values()[i] = 0.0;
    }
    auto u = vec::create(exec, gko::dim<2>(discretization_points, 1));
    u->copy_from(uh.get());
    auto u2 = vec::create(exec, gko::dim<2>(discretization_points, 1));
    u2->copy_from(u.get());

    const RealValueType reduction_factor{1e-7};
    auto mat = StencilMatrix<ValueType>::create(exec, discretization_points, -1,
                                                2, -1);
    auto mat2 = StencilMatrix<ValueType>::create(exec, discretization_points,
                                                 -1, 2, -1);

    for (auto i = 0; i < 3; ++i) {
        mat->apply(lend(rhs), lend(u));
    }

    std::chrono::nanoseconds time(0);
    std::chrono::nanoseconds time1(0);
    std::chrono::nanoseconds time2(0);
    auto tic = std::chrono::steady_clock::now();
    for (auto i = 0; i < num_reps; ++i) {
        mat->apply(lend(rhs), lend(u));
    }
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    std::shared_ptr<gko::AsyncHandle> handle;
    std::shared_ptr<gko::AsyncHandle> handle2;
    auto tic1 = std::chrono::steady_clock::now();
    for (auto i = 0; i < num_reps; ++i) {
        handle = mat->apply(lend(rhs), lend(u2), exec->get_handle_at(0));
        handle2 = mat2->apply(lend(rhs2), lend(u2), exec->get_handle_at(1));
    }
    auto toc1 = std::chrono::steady_clock::now();
    handle->wait();
    handle2->wait();
    auto toc2 = std::chrono::steady_clock::now();
    time1 += std::chrono::duration_cast<std::chrono::nanoseconds>(toc1 - tic1);
    time2 += std::chrono::duration_cast<std::chrono::nanoseconds>(toc2 - tic1);

    std::cout << "\nApply complete."
              << "\nMatrix size: " << mat->get_size()
              << "\n Total time (sync) (ms):"
              << static_cast<double>(time.count() / num_reps) / 1000000.0
              << "\n Total time  (async) (call only) (ms):"
              << static_cast<double>(time1.count() / num_reps) / 1000000.0
              << "\n Total time  (async) (call + run) (ms):"
              << static_cast<double>(time2.count() / num_reps) / 1000000.0
              << std::endl;
}
