// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/memory.hpp"

#include <new>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


// 128B: Intel L2 prefetchers pair cache lines; M1/POWER8 lines are 128B.
#ifdef __cpp_aligned_new
constexpr std::align_val_t cpu_allocator_alignment{128};
#endif


void* CpuAllocator::allocate(size_type num_bytes)
{
    // Apple Clang disables aligned new for macOS < 10.14.
#ifdef __cpp_aligned_new
    auto ptr =
        ::operator new (num_bytes, cpu_allocator_alignment, std::nothrow_t{});
#else
    auto ptr = ::operator new (num_bytes, std::nothrow_t{});
#endif
    GKO_ENSURE_ALLOCATED(ptr, "cpu", num_bytes);
    return ptr;
}


void CpuAllocator::deallocate(void* ptr)
{
#ifdef __cpp_aligned_new
    ::operator delete (ptr, cpu_allocator_alignment, std::nothrow_t{});
#else
    ::operator delete (ptr, std::nothrow_t{});
#endif
}


}  // namespace gko
