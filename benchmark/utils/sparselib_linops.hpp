// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_
#define GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>


#define IMPL_CREATE_SPARSELIB_LINOP(_type, ...)                              \
    template <>                                                              \
    std::unique_ptr<gko::LinOp> create_sparselib_linop<_type>(               \
        std::shared_ptr<const gko::Executor> exec)                           \
    {                                                                        \
        return __VA_ARGS__::create(exec);                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define STUB_CREATE_SPARSELIB_LINOP(_type)                     \
    template <>                                                \
    std::unique_ptr<gko::LinOp> create_sparselib_linop<_type>( \
        std::shared_ptr<const gko::Executor> exec) GKO_NOT_IMPLEMENTED


class cusparse_csr;
class cusparse_csrmp;
class cusparse_csrmm;
class cusparse_hybrid;
class cusparse_coo;
class cusparse_ell;
class cusparse_gcsr;
class cusparse_gcoo;
class cusparse_csrex;
class cusparse_gcsr;
class cusparse_gcsr2;
class cusparse_gcoo;


class hipsparse_csr;
class hipsparse_csrmm;
class hipsparse_hybrid;
class hipsparse_coo;
class hipsparse_ell;


class onemkl_csr;
class onemkl_optimized_csr;


template <typename OpTagType>
std::unique_ptr<gko::LinOp> create_sparselib_linop(
    std::shared_ptr<const gko::Executor> exec);


#endif  // GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_
