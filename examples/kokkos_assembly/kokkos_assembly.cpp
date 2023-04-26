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

#include <iostream>
#include <map>
#include <string>


#include <omp.h>
#include <Kokkos_Core.hpp>
#include <ginkgo/ginkgo.hpp>


template <typename T, typename MemorySpace>
struct kokkos_data;


template <typename ValueType, typename MemorySpace>
struct kokkos_data<gko::array<ValueType>, MemorySpace> {
    kokkos_data(gko::array<ValueType>& arr)
        : view(arr.get_data(), arr.get_num_elems())
    {}

    kokkos_data(ValueType* data, gko::size_type num_elements)
        : view(data, num_elements)
    {}

    template <typename... IntType>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(
        const IntType&... indices) const
    {
        return view(indices...);
    }

    Kokkos::View<ValueType*, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        view;
};

template <typename ValueType, typename MemorySpace>
struct kokkos_data<gko::matrix::Dense<ValueType>, MemorySpace> {
    kokkos_data(gko::matrix::Dense<ValueType>& mtx)
        : values(mtx.get_values(), mtx.get_size()[0], mtx.get_size()[1])
    {}

    Kokkos::View<ValueType**, Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        values;
};


template <typename ValueType, typename MemorySpace>
struct kokkos_data<const gko::matrix::Dense<ValueType>, MemorySpace> {
    kokkos_data(const gko::matrix::Dense<ValueType>& mtx)
        : values(mtx.get_const_values(), mtx.get_size()[0], mtx.get_size()[1])
    {}

    Kokkos::View<const ValueType**, Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        values;
};


template <typename ValueType, typename IndexType, typename MemorySpace>
struct kokkos_data<gko::device_matrix_data<ValueType, IndexType>, MemorySpace> {
    kokkos_data(gko::device_matrix_data<ValueType, IndexType>& md)
        : row_idxs(md.get_row_idxs(), md.get_num_elems()),
          col_idxs(md.get_col_idxs(), md.get_num_elems()),
          values(md.get_values(), md.get_num_elems())
    {}

    kokkos_data<gko::array<IndexType>, MemorySpace> row_idxs;
    kokkos_data<gko::array<IndexType>, MemorySpace> col_idxs;
    kokkos_data<gko::array<ValueType>, MemorySpace> values;
};


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
kokkos_data<T, MemorySpace> to_kokkos_data(T& data, MemorySpace ms = {})
{
    return kokkos_data<T, MemorySpace>{data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
kokkos_data<const T, MemorySpace> to_kokkos_data(const T& data,
                                                 MemorySpace ms = {})
{
    return kokkos_data<const T, MemorySpace>{data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
kokkos_data<T, MemorySpace> to_kokkos_data(T* data, MemorySpace ms = {})
{
    return kokkos_data<T, MemorySpace>{*data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
kokkos_data<const T, MemorySpace> to_kokkos_data(const T* data,
                                                 MemorySpace ms = {})
{
    return kokkos_data<const T, MemorySpace>{*data};
}


template <typename ExecType>
struct to_kokkos_execution_space;


template <typename ExecType>
struct to_kokkos_memory_space;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct to_kokkos_memory_space<gko::ReferenceExecutor> {
    using type = Kokkos::HostSpace;
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct to_kokkos_memory_space<gko::OmpExecutor> {
    using type = Kokkos::HostSpace;
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct to_kokkos_memory_space<gko::CudaExecutor> {
    using type = Kokkos::CudaSpace;
};
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
struct to_kokkos_memory_space<gko::HipError> {
    using type = Kokkos::SYCLDeviceUSMSpace;
};
#endif

#ifdef KOKKOS_ENABLE_SYCL
template <>
struct to_kokkos_memory_space<gko::DpcppExecutor> {};
#endif

// template <typename Closure, typename... Args>
// struct KokkosOperation : gko::Operation {
//     KokkosOperation(Closure&& op, Args&&... args)
//         : op_(std::forward<Closure>(op)),
//         args(std::forward_as_tuple(args...))
//     {}
//
//     void run(std::shared_ptr<const gko::ReferenceExecutor> exec) const
//     override
//     {
// #ifdef KOKKOS_ENABLE_SERIAL
//         apply(exec);
// #endif
//     }
//
//     void run(std::shared_ptr<const gko::OmpExecutor> exec) const override
//     {
//         apply(exec);
//     }
//
//     void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
//     {
//         apply(exec);
//     }
//
//     void run(std::shared_ptr<const gko::HipExecutor> exec) const override
//     {
//         apply(exec);
//     }
//
//     void run(std::shared_ptr<const gko::DpcppExecutor> exec) const override
//     {
//         apply(exec);
//     }
//
// private:
//     template <typename Space, std::size_t... I>
//     void apply_impl(Space space, std::index_sequence<I...>)
//     {
//         op_(Space::execution_space(),
//             to_kokkos_data<Space>(
//                 std::get<I>(std::forward<std::tuple<Args...>>(args)))...);
//     }
//
//     template <typename ExecType>
//     void apply(std::shared_ptr<const ExecType> execs)
//     {
//         using memspace = typename to_kokkos_memory_space<ExecType>::type;
//         apply_impl(memspace{}, std::make_index_sequence<sizeof...(Args)>{});
//     }
//
//     Closure op_;
//     std::tuple<Args...> args;
// };
//
//
// template <typename Fn, typename... Args>
// KokkosOperation<Fn, Args...> make_kokkos_kernel(Fn&& fn, Args&&... args)
//{
//     return {std::forward<Fn>(fn), std::forward<Args>(args)...};
// }

template <typename Closure, typename... Args>
struct kokkos_operator {
    using tuple_type =
        std::tuple<kokkos_data<typename std::remove_reference<Args>::type,
                               Kokkos::DefaultExecutionSpace::memory_space>...>;

    kokkos_operator(Closure&& op, Args&&... args)
        : fn(std::forward<Closure>(op)), args(to_kokkos_data(args)...)
    {}

    template <typename... ExecPolicyHandles>
    KOKKOS_INLINE_FUNCTION void operator()(ExecPolicyHandles&&... handles) const
    {
        apply_impl<ExecPolicyHandles...>(
            std::forward<ExecPolicyHandles>(handles)...,
            std::make_index_sequence<std::tuple_size<decltype(args)>::value>{});
    }


    template <typename... ExecPolicyHandles, std::size_t... I>
    KOKKOS_INLINE_FUNCTION void apply_impl(ExecPolicyHandles&&... handles,
                                           std::index_sequence<I...>) const
    {
        fn(std::forward<ExecPolicyHandles>(handles)...,
           std::get<I>(std::forward<tuple_type>(args))...);
    }

    Closure fn;
    mutable tuple_type args;
};

template <typename Closure, typename... Args>
kokkos_operator<Closure, Args...> make_kokkos_operator(Closure&& cl,
                                                       Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}


// Creates a stencil matrix in CSR format for the given number of discretization
// points.
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(gko::matrix::Csr<ValueType, IndexType>* matrix)
{
    auto exec = matrix->get_executor();
    const auto discretization_points = matrix->get_size()[0];

    // Over-allocate storage for the matrix elements. Each row has 3 entries,
    // except for the first and last one. To handle each row uniformly, we
    // allocate space for 3x the number of rows.
    gko::device_matrix_data<ValueType, IndexType> md(exec, matrix->get_size(),
                                                     discretization_points * 3);

    // Create the matrix entries. This also creates zero entries for the
    // first and second row to handle all rows uniformly.
    Kokkos::parallel_for(
        "generate_stencil_matrix", md.get_num_elems(),
        make_kokkos_operator(
            [discretization_points] __device__(int i, auto kokkos_md) {
                const ValueType coefs[] = {-1, 2, -1};
                auto ofs = static_cast<IndexType>((i % 3) - 1);
                auto row = static_cast<IndexType>(i / 3);
                auto col = row + ofs;

                // To prevent branching, a mask is used to set the entry to
                // zero, if the column is out-of-bounds
                auto mask = static_cast<IndexType>(0 <= col &&
                                                   col < discretization_points);

                kokkos_md.row_idxs(i) = mask * row;
                kokkos_md.col_idxs(i) = mask * col;
                kokkos_md.values(i) = mask * coefs[ofs + 1];
            },
            md));

    // Add up duplicate (zero) entries.
    md.sum_duplicates();

    // Build Csr matrix.
    matrix->read(std::move(md));
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ValueType>
void generate_rhs(Closure&& f, ValueType u0, ValueType u1,
                  gko::matrix::Dense<ValueType>* rhs)
{
    const auto discretization_points = rhs->get_size()[0];
    auto values = rhs->get_values();
    Kokkos::View<ValueType*> values_view(values, discretization_points);
    Kokkos::parallel_for(
        "generate_rhs", discretization_points, KOKKOS_LAMBDA(int i) {
            const ValueType h = 1.0 / (discretization_points + 1);
            const ValueType xi = ValueType(i + 1) * h;
            values_view(i) = -f(xi) * h * h;
            if (i == 0) {
                values_view(i) += u0;
            }
            if (i == discretization_points - 1) {
                values_view(i) += u1;
            }
        });
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType>
double calculate_error(int discretization_points,
                       const gko::matrix::Dense<ValueType>* u,
                       Closure&& correct_u)
{
    auto kokkos_u = to_kokkos_data(u);
    auto view = kokkos_u.values;
    auto error = 0.0;
    Kokkos::parallel_reduce(
        "calculate_error", discretization_points,
        KOKKOS_LAMBDA(int i, double& lsum) {
            const auto h = 1.0 / (discretization_points + 1);
            const auto xi = (i + 1) * h;
            lsum += Kokkos::Experimental::abs(
                (kokkos_u.values(i, 0) - correct_u(xi)) /
                Kokkos::Experimental::abs(correct_u(xi)));
        },
        error);
    return error;
}


int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType>;

    // Print help message. For details on the kokkos-options see
    // https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html#initialization-by-command-line-arguments
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [discretization_points] [kokkos-options]" << std::endl;
        std::exit(-1);
    }

    const unsigned int discretization_points =
        argc >= 2 ? std::atoi(argv[1]) : 100u;

    // chooses the executor that corresponds to the Kokkos DefaultExecutionSpace
    auto exec = []() -> std::shared_ptr<gko::Executor> {
#ifdef KOKKOS_ENABLE_SERIAL
        if (std::is_same<Kokkos::DefaultExecutionSpace,
                         Kokkos::Serial>::value) {
            return gko::ReferenceExecutor::create();
        }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        if (std::is_same<Kokkos::DefaultExecutionSpace,
                         Kokkos::OpenMP>::value) {
            return gko::OmpExecutor::create();
        }
#endif
#ifdef KOKKOS_ENABLE_CUDA
        if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value) {
            return gko::CudaExecutor::create(0,
                                             gko::ReferenceExecutor::create());
        }
#endif
#ifdef KOKKOS_ENABLE_HIP
        if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::HIP>::value) {
            return gko::HipExecutor::create(0,
                                            gko::ReferenceExecutor::create());
        }
#endif
    }();

    // problem:
    auto correct_u = [] KOKKOS_FUNCTION(ValueType x) { return x * x * x; };
    auto f = [] KOKKOS_FUNCTION(ValueType x) { return ValueType{6} * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // initialize vectors
    auto rhs = vec::create(exec, gko::dim<2>(discretization_points, 1));
    generate_rhs(f, u0, u1, rhs.get());
    auto u = vec::create(exec, gko::dim<2>(discretization_points, 1));
    for (int i = 0; i < u->get_size()[0]; ++i) {
        u->get_values()[i] = 0.0;
    }

    // initialize the stencil matrix
    auto A = share(mtx::create(
        exec, gko::dim<2>{discretization_points, discretization_points}));
    generate_stencil_matrix(A.get());

    const RealValueType reduction_factor{1e-7};
    // Generate solver and solve the system
    cg::build()
        .with_criteria(gko::stop::Iteration::build()
                           .with_max_iters(discretization_points)
                           .on(exec),
                       gko::stop::ResidualNorm<ValueType>::build()
                           .with_reduction_factor(reduction_factor)
                           .on(exec))
        .with_preconditioner(bj::build().on(exec))
        .on(exec)
        ->generate(A)
        ->apply(rhs, u);

    std::cout << "\nSolve complete."
              << "\nThe average relative error is "
              << calculate_error(discretization_points, u.get(), correct_u) /
                     discretization_points
              << std::endl;
}
