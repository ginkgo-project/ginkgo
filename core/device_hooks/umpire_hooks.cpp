// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <utility>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/memory.hpp>


namespace gko {


// The destructor needs `data` complete to instantiate the unique_ptr deleter.
struct UmpireAllocatorBase::data {};


UmpireAllocatorBase::UmpireAllocatorBase(std::string) GKO_NOT_COMPILED(umpire);


UmpireAllocatorBase::~UmpireAllocatorBase() = default;


const std::string& UmpireAllocatorBase::get_name() const
    GKO_NOT_COMPILED(umpire);


size_type UmpireAllocatorBase::get_current_size() const
    GKO_NOT_COMPILED(umpire);


size_type UmpireAllocatorBase::get_actual_size() const GKO_NOT_COMPILED(umpire);


size_type UmpireAllocatorBase::get_high_watermark() const
    GKO_NOT_COMPILED(umpire);


size_type UmpireAllocatorBase::get_allocation_count() const
    GKO_NOT_COMPILED(umpire);


void UmpireAllocatorBase::release() GKO_NOT_COMPILED(umpire);


void* UmpireAllocatorBase::allocate_impl(size_type) GKO_NOT_COMPILED(umpire);


void UmpireAllocatorBase::deallocate_impl(void*) noexcept {}


bool UmpireAllocatorBase::is_host_accessible() const noexcept { return false; }


bool UmpireAllocatorBase::is_device_accessible() const noexcept
{
    return false;
}


bool UmpireAllocatorBase::matches_device(int) const noexcept { return false; }


UmpireCpuAllocator::UmpireCpuAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{}


void* UmpireCpuAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireCpuAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


UmpireCudaAllocator::UmpireCudaAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{}


void* UmpireCudaAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireCudaAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


bool UmpireCudaAllocator::check_environment(int, CUstream_st*) const
{
    return false;
}


UmpireHipAllocator::UmpireHipAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{}


void* UmpireHipAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireHipAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


bool UmpireHipAllocator::check_environment(int, GKO_HIP_STREAM_STRUCT*) const
{
    return false;
}


std::shared_ptr<UmpireCpuAllocator> make_umpire_cpu_pool(std::string,
                                                         umpire_pool_config)
    GKO_NOT_COMPILED(umpire);


std::shared_ptr<UmpireCudaAllocator> make_umpire_cuda_pool(std::string,
                                                           umpire_pool_config)
    GKO_NOT_COMPILED(umpire);


std::shared_ptr<UmpireHipAllocator> make_umpire_hip_pool(std::string,
                                                         umpire_pool_config)
    GKO_NOT_COMPILED(umpire);


}  // namespace gko
