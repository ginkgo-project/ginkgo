// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_
#define GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_


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


#endif  // GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_
