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
