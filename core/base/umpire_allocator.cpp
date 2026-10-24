// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>
#include <umpire/Tracking.hpp>
#include <umpire/util/error.hpp>
#include <umpire/util/MemoryResourceTraits.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/memory.hpp>


namespace gko {
namespace {


using resource_type = umpire::MemoryResourceTraits::resource_type;


bool is_host_accessible_resource(resource_type r) noexcept
{
    return r == resource_type::host || r == resource_type::pinned ||
           r == resource_type::um;
}


bool is_device_accessible_resource(resource_type r) noexcept
{
    return r == resource_type::device || r == resource_type::pinned ||
           r == resource_type::um;
}


// getAllocator() creates resources such as "DEVICE::1" lazily and can throw;
// convert so that only Ginkgo exceptions escape.
umpire::Allocator get_umpire_allocator(umpire::ResourceManager& rm,
                                       const std::string& name)
{
    try {
        return rm.getAllocator(name);
    } catch (const umpire::runtime_error& e) {
        throw Error(
            __FILE__, __LINE__,
            "Could not get Umpire allocator '" + name + "': " + e.what());
    }
}


}  // namespace


struct UmpireAllocatorBase::data {
    std::string name;
    umpire::Allocator alloc;
    umpire::MemoryResourceTraits traits;
};


UmpireAllocatorBase::UmpireAllocatorBase(std::string name)
    : data_{std::make_unique<data>()}
{
    auto& rm = umpire::ResourceManager::getInstance();
    // isAllocator rather than tryGetAllocator, which needs Umpire >=
    // v2026.07.0.
    if (!rm.isAllocator(name)) {
        throw Error(__FILE__, __LINE__,
                    "No Umpire allocator named '" + name + "'");
    }
    data_->name = std::move(name);
    data_->alloc = get_umpire_allocator(rm, data_->name);
    data_->traits = data_->alloc.getAllocationStrategy()->getTraits();
}


UmpireAllocatorBase::~UmpireAllocatorBase() = default;


const std::string& UmpireAllocatorBase::get_name() const { return data_->name; }


size_type UmpireAllocatorBase::get_current_size() const
{
    return static_cast<size_type>(data_->alloc.getCurrentSize());
}


size_type UmpireAllocatorBase::get_actual_size() const
{
    return static_cast<size_type>(data_->alloc.getActualSize());
}


size_type UmpireAllocatorBase::get_high_watermark() const
{
    return static_cast<size_type>(data_->alloc.getHighWatermark());
}


size_type UmpireAllocatorBase::get_allocation_count() const
{
    return static_cast<size_type>(data_->alloc.getAllocationCount());
}


void UmpireAllocatorBase::release() { data_->alloc.release(); }


void* UmpireAllocatorBase::allocate_impl(size_type num_bytes)
{
    void* ptr{};
    try {
        ptr = data_->alloc.allocate(static_cast<std::size_t>(num_bytes));
    } catch (const umpire::runtime_error&) {
        throw AllocationError(__FILE__, __LINE__, "umpire:" + this->get_name(),
                              num_bytes);
    }
    // Not GKO_ENSURE_ALLOCATED: it builds its name string even on success.
    if (ptr == nullptr) {
        throw AllocationError(__FILE__, __LINE__, "umpire:" + this->get_name(),
                              num_bytes);
    }
    return ptr;
}


void UmpireAllocatorBase::deallocate_impl(void* ptr) noexcept
{
    // Executor::raw_free is noexcept, so exit like GKO_EXIT_ON_CUDA_ERROR.
    try {
        data_->alloc.deallocate(ptr);
    } catch (const std::exception& e) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr << "Unrecoverable Umpire error in " << __func__ << ":"
                  << __LINE__ << ": " << e.what() << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(1);
    } catch (...) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr << "Unrecoverable Umpire error in " << __func__ << ":"
                  << __LINE__ << ": unknown exception" << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(1);
    }
}


bool UmpireAllocatorBase::is_host_accessible() const noexcept
{
    return is_host_accessible_resource(data_->traits.resource);
}


bool UmpireAllocatorBase::is_device_accessible() const noexcept
{
    return is_device_accessible_resource(data_->traits.resource);
}


bool UmpireAllocatorBase::matches_device(int device_id) const noexcept
{
    if (data_->traits.resource != resource_type::device) {
        return true;
    }
    return data_->traits.id == device_id;
}


namespace {


// Creates the pool unless `name` is already registered; the caller wraps it.
void ensure_umpire_pool(const std::string& name,
                        const std::string& default_resource,
                        const umpire_pool_config& config,
                        bool (*accepts)(resource_type) noexcept,
                        const char* accessible_from)
{
    auto& rm = umpire::ResourceManager::getInstance();
    if (rm.isAllocator(name)) {
        return;
    }
    if (config.alignment == 0 ||
        (config.alignment & (config.alignment - 1)) != 0) {
        // Umpire's AlignedAllocation does not validate alignment.
        throw Error(__FILE__, __LINE__,
                    "Umpire pool '" + name +
                        "': alignment must be a nonzero power of two, got " +
                        std::to_string(config.alignment));
    }
    const auto resource =
        config.resource.empty() ? default_resource : config.resource;
    if (!rm.isAllocator(resource)) {
        throw Error(__FILE__, __LINE__,
                    "No Umpire memory resource named '" + resource +
                        "' for pool '" + name + "'");
    }
    auto upstream = get_umpire_allocator(rm, resource);
    if (!accepts(upstream.getAllocationStrategy()->getTraits().resource)) {
        // Checked before registering: a registered pool stays registered, so
        // a corrected retry would otherwise reuse the wrong pool.
        throw Error(__FILE__, __LINE__,
                    "Umpire memory resource '" + resource + "' for pool '" +
                        name + "' is not " + accessible_from);
    }
    const auto tracking = config.tracked ? umpire::Tracking::Tracked
                                         : umpire::Tracking::Untracked;

    const auto pool_name = config.thread_safe ? name + "_pool" : name;
    try {
        rm.makeAllocator<umpire::strategy::QuickPool>(
            pool_name, tracking, upstream,
            static_cast<std::size_t>(config.initial_block_size),
            static_cast<std::size_t>(config.growth_block_size),
            static_cast<std::size_t>(config.alignment));
    } catch (const umpire::runtime_error& e) {
        throw Error(
            __FILE__, __LINE__,
            "Could not create Umpire pool '" + pool_name + "': " + e.what());
    }

    if (config.thread_safe) {
        try {
            rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
                name, tracking, rm.getAllocator(pool_name));
        } catch (const umpire::runtime_error& e) {
            // pool_name stays registered, so a retry fails differently.
            throw Error(__FILE__, __LINE__,
                        "Could not create Umpire pool '" + name + "': " +
                            e.what() + " (the underlying pool '" + pool_name +
                            "' was already created and remains registered with "
                            "Umpire)");
        }
    }
}


}  // namespace


UmpireCpuAllocator::UmpireCpuAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{
    if (!this->is_host_accessible()) {
        throw Error(__FILE__, __LINE__,
                    "Umpire allocator '" + this->get_name() +
                        "' is not host-accessible");
    }
}


void* UmpireCpuAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireCpuAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


UmpireCudaAllocator::UmpireCudaAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{
    if (!this->is_device_accessible()) {
        throw Error(__FILE__, __LINE__,
                    "Umpire allocator '" + this->get_name() +
                        "' is not device-accessible");
    }
}


void* UmpireCudaAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireCudaAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


bool UmpireCudaAllocator::check_environment(int device_id, CUstream_st*) const
{
    return this->matches_device(device_id);
}


UmpireHipAllocator::UmpireHipAllocator(std::string name)
    : UmpireAllocatorBase{std::move(name)}
{
    if (!this->is_device_accessible()) {
        throw Error(__FILE__, __LINE__,
                    "Umpire allocator '" + this->get_name() +
                        "' is not device-accessible");
    }
}


void* UmpireHipAllocator::allocate(size_type num_bytes)
{
    return this->allocate_impl(num_bytes);
}


void UmpireHipAllocator::deallocate(void* ptr) { this->deallocate_impl(ptr); }


bool UmpireHipAllocator::check_environment(int device_id,
                                           GKO_HIP_STREAM_STRUCT*) const
{
    return this->matches_device(device_id);
}


std::shared_ptr<UmpireCpuAllocator> make_umpire_cpu_pool(
    std::string name, umpire_pool_config config)
{
    ensure_umpire_pool(name, "HOST", config, is_host_accessible_resource,
                       "host-accessible");
    return std::make_shared<UmpireCpuAllocator>(std::move(name));
}


std::shared_ptr<UmpireCudaAllocator> make_umpire_cuda_pool(
    std::string name, umpire_pool_config config)
{
    ensure_umpire_pool(name, "DEVICE", config, is_device_accessible_resource,
                       "device-accessible");
    return std::make_shared<UmpireCudaAllocator>(std::move(name));
}


std::shared_ptr<UmpireHipAllocator> make_umpire_hip_pool(
    std::string name, umpire_pool_config config)
{
    ensure_umpire_pool(name, "DEVICE", config, is_device_accessible_resource,
                       "device-accessible");
    return std::make_shared<UmpireHipAllocator>(std::move(name));
}


}  // namespace gko
