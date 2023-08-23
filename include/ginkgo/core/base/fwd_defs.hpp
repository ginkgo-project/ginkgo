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

#ifndef GKO_PUBLIC_CORE_BASE_FWD_DEFS_HPP_
#define GKO_PUBLIC_CORE_BASE_FWD_DEFS_HPP_


#include <ginkgo/config.hpp>


struct cublasContext;

struct cusparseContext;

struct CUstream_st;

struct CUevent_st;

struct hipblasContext;

struct hipsparseContext;

#if GINKGO_HIP_PLATFORM_HCC
struct ihipStream_t;
struct ihipEvent_t;
#define GKO_HIP_STREAM_STRUCT ihipStream_t
#define GKO_HIP_EVENT_STRUCT ihipEvent_t
#else
#define GKO_HIP_STREAM_STRUCT CUstream_st
#define GKO_HIP_EVENT_STRUCT CUevent_st
#endif


// after intel/llvm September'22 release, which uses major version 6, they
// introduce another inline namespace _V1.
#if GINKGO_DPCPP_MAJOR_VERSION >= 6
namespace sycl {
inline namespace _V1 {


class queue;
class event;


}  // namespace _V1
}  // namespace sycl
#else  // GINKGO_DPCPP_MAJOR_VERSION < 6
inline namespace cl {
namespace sycl {


class queue;
class event;


}  // namespace sycl
}  // namespace cl
#endif


#endif  // GKO_PUBLIC_CORE_BASE_FWD_DEFS_HPP_
