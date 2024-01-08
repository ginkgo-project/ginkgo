// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/memory.hpp>


#include <new>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


void* CpuAllocator::allocate(size_type num_bytes)
{
    auto ptr = ::operator new (num_bytes, std::nothrow_t{});
    GKO_ENSURE_ALLOCATED(ptr, "cpu", num_bytes);
    return ptr;
}


void CpuAllocator::deallocate(void* ptr)
{
    ::operator delete (ptr, std::nothrow_t{});
}


}  // namespace gko
