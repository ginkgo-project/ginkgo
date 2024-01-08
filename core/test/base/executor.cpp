// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <thread>
#include <type_traits>


#if defined(__unix__) || defined(__APPLE__)
#include <utmpx.h>
#endif


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/memory.hpp>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


TEST(OmpExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr omp = gko::OmpExecutor::create();
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = omp->alloc<int>(num_elems));
    ASSERT_NO_THROW(omp->free(ptr));
}


TEST(OmpExecutor, FreeAcceptsNullptr)
{
    exec_ptr omp = gko::OmpExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
}


TEST(OmpExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr omp = gko::OmpExecutor::create();
    int* ptr = nullptr;

    ASSERT_THROW(ptr = omp->alloc<int>(num_elems), gko::AllocationError);

    omp->free(ptr);
}


TEST(OmpExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    int* copy = omp->alloc<int>(num_elems);

    // user code is run on the OMP, so local variables are in OMP memory
    omp->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    omp->free(copy);
}


TEST(OmpExecutor, IsItsOwnMaster)
{
    exec_ptr omp = gko::OmpExecutor::create();

    ASSERT_EQ(omp, omp->get_master());
}


#if GKO_HAVE_HWLOC


inline int get_os_id(int log_id)
{
    return gko::machine_topology::get_instance()->get_core(log_id)->os_id;
}


TEST(MachineTopology, CanBindToASpecificCore)
{
    auto cpu_sys = sched_getcpu();

    const int bind_core = 3;
    gko::machine_topology::get_instance()->bind_to_cores(
        std::vector<int>{bind_core});

    cpu_sys = sched_getcpu();
    ASSERT_EQ(cpu_sys, get_os_id(bind_core));
}


TEST(MachineTopology, CanBindToARangeofCores)
{
    auto cpu_sys = sched_getcpu();

    const std::vector<int> bind_core = {1, 3};
    gko::machine_topology::get_instance()->bind_to_cores(bind_core);

    cpu_sys = sched_getcpu();
    ASSERT_TRUE(cpu_sys == get_os_id(3) || cpu_sys == get_os_id(1));
}


#endif


TEST(ReferenceExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = ref->alloc<int>(num_elems));
    ASSERT_NO_THROW(ref->free(ptr));
}


TEST(ReferenceExecutor, FreeAcceptsNullptr)
{
    exec_ptr omp = gko::ReferenceExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
}


TEST(ReferenceExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* ptr = nullptr;

    ASSERT_THROW(ptr = ref->alloc<int>(num_elems), gko::AllocationError);

    ref->free(ptr);
}


TEST(ReferenceExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesSingleValue)
{
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* el = ref->alloc<int>(1);
    el[0] = 83683;

    EXPECT_EQ(83683, ref->copy_val_to_host(el));

    ref->free(el);
}


TEST(ReferenceExecutor, CopiesDataFromOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy_from(omp, num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataToOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* copy = omp->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    omp->copy_from(ref, num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = gko::ReferenceExecutor::create();

    ASSERT_EQ(ref, ref->get_master());
}


TEST(CudaExecutor, KnowsItsMaster)
{
    auto ref = gko::ReferenceExecutor::create();
    exec_ptr cuda = gko::CudaExecutor::create(0, ref);

    ASSERT_EQ(ref, cuda->get_master());
}


TEST(CudaExecutor, KnowsItsDeviceId)
{
    auto ref = gko::ReferenceExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, ref);

    ASSERT_EQ(0, cuda->get_device_id());
}


TEST(HipExecutor, KnowsItsMaster)
{
    auto ref = gko::ReferenceExecutor::create();
    exec_ptr hip = gko::HipExecutor::create(0, ref);

    ASSERT_EQ(ref, hip->get_master());
}


TEST(HipExecutor, KnowsItsDeviceId)
{
    auto ref = gko::ReferenceExecutor::create();
    auto hip = gko::HipExecutor::create(0, ref);

    ASSERT_EQ(0, hip->get_device_id());
}


TEST(DpcppExecutor, KnowsItsMaster)
{
    auto ref = gko::ReferenceExecutor::create();
    exec_ptr dpcpp = gko::DpcppExecutor::create(0, ref);

    ASSERT_EQ(ref, dpcpp->get_master());
}


TEST(DpcppExecutor, KnowsItsDeviceId)
{
    auto ref = gko::ReferenceExecutor::create();
    auto dpcpp = gko::DpcppExecutor::create(0, ref);

    ASSERT_EQ(0, dpcpp->get_device_id());
}


TEST(Executor, CanVerifyMemory)
{
    auto ref = gko::ReferenceExecutor::create();
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, ref);
    auto cuda = gko::CudaExecutor::create(0, ref);
    auto omp2 = gko::OmpExecutor::create();
    auto hip2 = gko::HipExecutor::create(0, ref);
    auto cuda2 = gko::CudaExecutor::create(0, ref);
    auto hip_1 = gko::HipExecutor::create(1, ref);
    auto cuda_1 = gko::CudaExecutor::create(1, ref);
    std::shared_ptr<gko::DpcppExecutor> host_dpcpp;
    std::shared_ptr<gko::DpcppExecutor> cpu_dpcpp;
    std::shared_ptr<gko::DpcppExecutor> gpu_dpcpp;
    std::shared_ptr<gko::DpcppExecutor> host_dpcpp_dup;
    std::shared_ptr<gko::DpcppExecutor> cpu_dpcpp_dup;
    std::shared_ptr<gko::DpcppExecutor> gpu_dpcpp_dup;
    if (gko::DpcppExecutor::get_num_devices("host")) {
        host_dpcpp = gko::DpcppExecutor::create(0, ref, "host");
        host_dpcpp_dup = gko::DpcppExecutor::create(0, ref, "host");
    }
    if (gko::DpcppExecutor::get_num_devices("cpu")) {
        cpu_dpcpp = gko::DpcppExecutor::create(0, ref, "cpu");
        cpu_dpcpp_dup = gko::DpcppExecutor::create(0, ref, "cpu");
    }
    if (gko::DpcppExecutor::get_num_devices("gpu")) {
        gpu_dpcpp = gko::DpcppExecutor::create(0, ref, "gpu");
        gpu_dpcpp_dup = gko::DpcppExecutor::create(0, ref, "gpu");
    }

    ASSERT_EQ(false, ref->memory_accessible(omp));
    ASSERT_EQ(false, omp->memory_accessible(ref));
    ASSERT_EQ(false, ref->memory_accessible(hip));
    ASSERT_EQ(false, hip->memory_accessible(ref));
    ASSERT_EQ(false, omp->memory_accessible(hip));
    ASSERT_EQ(false, hip->memory_accessible(omp));
    ASSERT_EQ(false, ref->memory_accessible(cuda));
    ASSERT_EQ(false, cuda->memory_accessible(ref));
    ASSERT_EQ(false, omp->memory_accessible(cuda));
    ASSERT_EQ(false, cuda->memory_accessible(omp));
    if (gko::DpcppExecutor::get_num_devices("host")) {
        ASSERT_EQ(false, host_dpcpp->memory_accessible(ref));
        ASSERT_EQ(false, ref->memory_accessible(host_dpcpp));
        ASSERT_EQ(true, host_dpcpp->memory_accessible(omp));
        ASSERT_EQ(true, omp->memory_accessible(host_dpcpp));
        ASSERT_EQ(true, host_dpcpp->memory_accessible(host_dpcpp_dup));
        ASSERT_EQ(true, host_dpcpp_dup->memory_accessible(host_dpcpp));
    }
    if (gko::DpcppExecutor::get_num_devices("cpu")) {
        ASSERT_EQ(false, ref->memory_accessible(cpu_dpcpp));
        ASSERT_EQ(false, cpu_dpcpp->memory_accessible(ref));
        ASSERT_EQ(true, cpu_dpcpp->memory_accessible(omp));
        ASSERT_EQ(true, omp->memory_accessible(cpu_dpcpp));
        ASSERT_EQ(true, cpu_dpcpp->memory_accessible(cpu_dpcpp_dup));
        ASSERT_EQ(true, cpu_dpcpp_dup->memory_accessible(cpu_dpcpp));
    }
    if (gko::DpcppExecutor::get_num_devices("gpu")) {
        ASSERT_EQ(false, gpu_dpcpp->memory_accessible(ref));
        ASSERT_EQ(false, ref->memory_accessible(gpu_dpcpp));
        ASSERT_EQ(false, gpu_dpcpp->memory_accessible(omp));
        ASSERT_EQ(false, omp->memory_accessible(gpu_dpcpp));
        ASSERT_EQ(false, gpu_dpcpp->memory_accessible(gpu_dpcpp_dup));
        ASSERT_EQ(false, gpu_dpcpp_dup->memory_accessible(gpu_dpcpp));
    }
#if GINKGO_HIP_PLATFORM_NVCC
    ASSERT_EQ(true, hip->memory_accessible(cuda));
    ASSERT_EQ(true, cuda->memory_accessible(hip));
    ASSERT_EQ(true, hip_1->memory_accessible(cuda_1));
    ASSERT_EQ(true, cuda_1->memory_accessible(hip_1));
#else
    ASSERT_EQ(false, hip->memory_accessible(cuda));
    ASSERT_EQ(false, cuda->memory_accessible(hip));
    ASSERT_EQ(false, hip_1->memory_accessible(cuda_1));
    ASSERT_EQ(false, cuda_1->memory_accessible(hip_1));
#endif
    ASSERT_EQ(true, omp->memory_accessible(omp2));
    ASSERT_EQ(true, hip->memory_accessible(hip2));
    ASSERT_EQ(true, cuda->memory_accessible(cuda2));
    ASSERT_EQ(false, hip->memory_accessible(hip_1));
    ASSERT_EQ(false, cuda->memory_accessible(hip_1));
    ASSERT_EQ(false, cuda->memory_accessible(cuda_1));
    ASSERT_EQ(false, hip->memory_accessible(cuda_1));
}


struct MockAllocator : gko::CpuAllocator {
    void deallocate(void* ptr) noexcept override
    {
        called_free = true;
        CpuAllocator::deallocate(ptr);
    }

    mutable bool called_free{false};
};


TEST(ExecutorDeleter, DeletesObject)
{
    auto alloc = std::make_shared<MockAllocator>();
    auto ref = gko::ReferenceExecutor::create(alloc);
    auto x = ref->alloc<int>(5);

    gko::executor_deleter<int>{ref}(x);

    ASSERT_TRUE(alloc->called_free);
}


TEST(ExecutorDeleter, AvoidsDeletionForNullExecutor)
{
    int x[5];

    ASSERT_NO_THROW(gko::executor_deleter<int>{nullptr}(x));
}


struct DummyLogger : public gko::log::Logger {
    DummyLogger()
        : gko::log::Logger(gko::log::Logger::executor_events_mask |
                           gko::log::Logger::operation_events_mask)
    {}

    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type& num_bytes) const override
    {
        allocation_started++;
    }
    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr& location) const override
    {
        allocation_completed++;
    }

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr& location) const override
    {
        free_started++;
    }


    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr& location) const override
    {
        free_completed++;
    }

    void on_copy_started(const gko::Executor* exec_from,
                         const gko::Executor* exec_to,
                         const gko::uintptr& loc_from,
                         const gko::uintptr& loc_to,
                         const gko::size_type& num_bytes) const override
    {
        copy_started++;
    }

    void on_copy_completed(const gko::Executor* exec_from,
                           const gko::Executor* exec_to,
                           const gko::uintptr& loc_from,
                           const gko::uintptr& loc_to,
                           const gko::size_type& num_bytes) const override
    {
        copy_completed++;
    }

    void on_operation_launched(const gko::Executor* exec,
                               const gko::Operation* op) const override
    {
        operation_launched++;
    }


    void on_operation_completed(const gko::Executor* exec,
                                const gko::Operation* op) const override
    {
        operation_completed++;
    }

    mutable int allocation_started = 0;
    mutable int allocation_completed = 0;
    mutable int free_started = 0;
    mutable int free_completed = 0;
    mutable int copy_started = 0;
    mutable int copy_completed = 0;
    mutable int operation_launched = 0;
    mutable int operation_completed = 0;
};


class ExecutorLogging : public ::testing::Test {
protected:
    ExecutorLogging()
        : exec(gko::ReferenceExecutor::create()),
          logger(std::make_shared<DummyLogger>())
    {
        exec->add_logger(logger);
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<DummyLogger> logger;
};


TEST_F(ExecutorLogging, LogsAllocationAndFree)
{
    auto before_logger = *logger;

    auto p = exec->alloc<int>(1);
    exec->free(p);

    ASSERT_EQ(logger->allocation_started, before_logger.allocation_started + 1);
    ASSERT_EQ(logger->allocation_completed,
              before_logger.allocation_completed + 1);
    ASSERT_EQ(logger->free_started, before_logger.free_started + 1);
    ASSERT_EQ(logger->free_completed, before_logger.free_completed + 1);
}


TEST_F(ExecutorLogging, LogsCopy)
{
    auto before_logger = *logger;

    exec->copy<std::nullptr_t>(0, nullptr, nullptr);

    ASSERT_EQ(logger->copy_started, before_logger.copy_started + 1);
    ASSERT_EQ(logger->copy_completed, before_logger.copy_completed + 1);
}


class ExampleOperation : public gko::Operation {
public:
    void run(std::shared_ptr<const gko::OmpExecutor>) const override {}
    void run(std::shared_ptr<const gko::CudaExecutor>) const override {}
    void run(std::shared_ptr<const gko::HipExecutor>) const override {}
    void run(std::shared_ptr<const gko::DpcppExecutor>) const override {}
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override {}
};


TEST_F(ExecutorLogging, LogsOperation)
{
    auto before_logger = *logger;

    exec->run(ExampleOperation());

    ASSERT_EQ(logger->operation_launched, before_logger.operation_launched + 1);
    ASSERT_EQ(logger->operation_completed,
              before_logger.operation_completed + 1);
}


}  // namespace
