/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_
#define GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_


#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


inline namespace cl {
namespace sycl {

class queue;

}  // namespace sycl
}  // namespace cl


struct cublasContext;

struct cusparseContext;

struct hipblasContext;

struct hipsparseContext;


namespace gko {


#define GKO_FORWARD_DECLARE(_type, ...) class _type

GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_FORWARD_DECLARE);

#undef GKO_FORWARD_DECLARE


class ReferenceExecutor;


namespace detail {


template <typename>
class ExecutorBase;


}  // namespace detail


/**
 * Operations can be used to define functionalities whose implementations differ
 * among devices.
 *
 * This is done by extending the Operation class and implementing the overloads
 * of the Operation::run() method for all Executor types. When invoking the
 * Executor::run() method with the Operation as input, the library will select
 * the Operation::run() overload corresponding to the dynamic type of the
 * Executor instance.
 *
 * Consider an overload of `operator<<` for Executors, which prints some basic
 * device information (e.g. device type and id) of the Executor to a C++ stream:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec);
 * ```
 *
 * One possible implementation would be to use RTTI to find the dynamic type of
 * the Executor, However, using the Operation feature of Ginkgo, there is a
 * more elegant approach which utilizes polymorphism. The first step is to
 * define an Operation that will print the desired information for each Executor
 * type.
 *
 * ```
 * class DeviceInfoPrinter : public gko::Operation {
 * public:
 *     explicit DeviceInfoPrinter(std::ostream &os) : os_(os) {}
 *
 *     void run(const gko::OmpExecutor *) const override { os_ << "OMP"; }
 *
 *     void run(const gko::CudaExecutor *exec) const override
 *     { os_ << "CUDA(" << exec->get_device_id() << ")"; }
 *
 *     void run(const gko::HipExecutor *exec) const override
 *     { os_ << "HIP(" << exec->get_device_id() << ")"; }
 *
 *     void run(const gko::DpcppExecutor *exec) const override
 *     { os_ << "DPC++(" << exec->get_device_id() << ")"; }
 *
 *     // This is optional, if not overloaded, defaults to OmpExecutor overload
 *     void run(const gko::ReferenceExecutor *) const override
 *     { os_ << "Reference CPU"; }
 *
 * private:
 *     std::ostream &os_;
 * };
 * ```
 *
 * Using DeviceInfoPrinter, the implementation of `operator<<` is as simple as
 * calling the run() method of the executor.
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     DeviceInfoPrinter printer(os);
 *     exec.run(printer);
 *     return os;
 * }
 * ```
 *
 * Now it is possible to write the following code:
 *
 * ```
 * auto omp = gko::OmpExecutor::create();
 * std::cout << *omp << std::endl
 *           << *gko::CudaExecutor::create(0, omp) << std::endl
 *           << *gko::HipExecutor::create(0, omp) << std::endl
 *           << *gko::DpcppExecutor::create(0, omp) << std::endl
 *           << *gko::ReferenceExecutor::create() << std::endl;
 * ```
 *
 * which produces the expected output:
 *
 * ```
 * OMP
 * CUDA(0)
 * HIP(0)
 * DPC++(0)
 * Reference CPU
 * ```
 *
 * One might feel that this code is too complicated for such a simple task.
 * Luckily, there is an overload of the Executor::run() method, which is
 * designed to facilitate writing simple operations like this one. The method
 * takes three closures as input: one which is run for OMP, one for CUDA
 * executors, one for HIP executors, and the last one for DPC++ executors. Using
 * this method, there is no need to implement an Operation subclass:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     exec.run(
 *         [&]() { os << "OMP"; },  // OMP closure
 *         [&]() { os << "CUDA("    // CUDA closure
 *                    << static_cast<gko::CudaExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; },
 *         [&]() { os << "HIP("    // HIP closure
 *                    << static_cast<gko::HipExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *         [&]() { os << "DPC++("    // DPC++ closure
 *                    << static_cast<gko::DpcppExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *     return os;
 * }
 * ```
 *
 * Using this approach, however, it is impossible to distinguish between
 * a OmpExecutor and ReferenceExecutor, as both of them call the OMP closure.
 *
 * @ingroup Executor
 */
class Operation {
public:
#define GKO_DECLARE_RUN_OVERLOAD(_type, ...) \
    virtual void run(std::shared_ptr<const _type>) const

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_RUN_OVERLOAD);

#undef GKO_DECLARE_RUN_OVERLOAD

    // ReferenceExecutor overload can be defaulted to OmpExecutor's
    virtual void run(std::shared_ptr<const ReferenceExecutor> executor) const;

    /**
     * Returns the operation's name.
     *
     * @return the operation's name
     */
    virtual const char *get_name() const noexcept;
};

#define GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(_type, _namespace, _kernel)    \
public:                                                                      \
    void run(std::shared_ptr<const ::gko::_type> exec) const override        \
    {                                                                        \
        this->call(counts{}, exec);                                          \
    }                                                                        \
                                                                             \
private:                                                                     \
    template <int... Ns>                                                     \
    void call(::gko::syn::value_list<int, Ns...>,                            \
              std::shared_ptr<const ::gko::_type> &exec) const               \
    {                                                                        \
        ::gko::kernels::_namespace::_kernel(                                 \
            exec, std::forward<Args>(std::get<Ns>(data))...);                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_DETAIL_DEFINE_RUN_OVERLOAD(_type, _namespace, _kernel, ...)      \
public:                                                                      \
    void run(std::shared_ptr<const ::gko::_type> exec) const override        \
    {                                                                        \
        this->call(counts{}, exec);                                          \
    }                                                                        \
                                                                             \
private:                                                                     \
    template <int... Ns>                                                     \
    void call(::gko::syn::value_list<int, Ns...>,                            \
              std::shared_ptr<const ::gko::_type> &exec) const               \
    {                                                                        \
        ::gko::kernels::_namespace::_kernel(                                 \
            exec, std::forward<Args>(std::get<Ns>(data))...);                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Binds a set of device-specific kernels to an Operation.
 *
 * It also defines a helper function which creates the associated operation.
 * Any input arguments passed to the helper function are forwarded to the
 * kernel when the operation is executed.
 *
 * The kernels used to bind the operation are searched in `kernels::DEV_TYPE`
 * namespace, where `DEV_TYPE` is replaced by `omp`, `cuda`, `hip`, `dpcpp` and
 * `reference`.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * // define the omp, cuda, hip and reference kernels which will be bound to the
 * // operation
 * namespace kernels {
 * namespace omp {
 * void my_kernel(int x) {
 *      // omp code
 * }
 * }
 * namespace cuda {
 * void my_kernel(int x) {
 *      // cuda code
 * }
 * }
 * namespace hip {
 * void my_kernel(int x) {
 *      // hip code
 * }
 * }
 * namespace dpcpp {
 * void my_kernel(int x) {
 *      // dpcpp code
 * }
 * }
 * namespace reference {
 * void my_kernel(int x) {
 *     // reference code
 * }
 * }
 *
 * // Bind the kernels to the operation
 * GKO_REGISTER_OPERATION(my_op, my_kernel);
 *
 * int main() {
 *     // create executors
 *     auto omp = OmpExecutor::create();
 *     auto cuda = CudaExecutor::create(omp, 0);
 *     auto hip = HipExecutor::create(omp, 0);
 *     auto dpcpp = DpcppExecutor::create(omp, 0);
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op(5); // x = 5
 *
 *     omp->run(op);  // run omp kernel
 *     cuda->run(op);  // run cuda kernel
 *     hip->run(op);  // run hip kernel
 *     dpcpp->run(op);  // run DPC++ kernel
 *     ref->run(op);  // run reference kernel
 * }
 * ```
 *
 * @ingroup Executor
 */
#define GKO_REGISTER_OPERATION(_name, _kernel)                                \
    template <typename... Args>                                               \
    class _name##_operation : public Operation {                              \
        using counts =                                                        \
            ::gko::syn::as_list<::gko::syn::range<0, sizeof...(Args)>>;       \
                                                                              \
    public:                                                                   \
        explicit _name##_operation(Args &&... args)                           \
            : data(std::forward<Args>(args)...)                               \
        {}                                                                    \
                                                                              \
        const char *get_name() const noexcept override                        \
        {                                                                     \
            static auto name = [this] {                                       \
                std::ostringstream oss;                                       \
                oss << #_kernel << '#' << sizeof...(Args);                    \
                return oss.str();                                             \
            }();                                                              \
            return name.c_str();                                              \
        }                                                                     \
                                                                              \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(OmpExecutor, omp, _kernel);     \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(CudaExecutor, cuda, _kernel);   \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(HipExecutor, hip, _kernel);     \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(DpcppExecutor, dpcpp, _kernel); \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(ReferenceExecutor, reference,   \
                                              _kernel);                       \
                                                                              \
    private:                                                                  \
        mutable std::tuple<Args &&...> data;                                  \
    };                                                                        \
                                                                              \
    template <typename... Args>                                               \
    static _name##_operation<Args...> make_##_name(Args &&... args)           \
    {                                                                         \
        return _name##_operation<Args...>(std::forward<Args>(args)...);       \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")


#define GKO_DECLARE_EXECUTOR_FRIEND(_type, ...) friend class _type

/**
 * The first step in using the Ginkgo library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * Ginkgo currently supports three different executor types:
 *
 * +    OmpExecutor specifies that the data should be stored and the associated
 *      operations executed on an OpenMP-supporting device (e.g. host CPU);
 * +    CudaExecutor specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    HipExecutor specifies that the data should be stored and the
 *      operations executed on either an NVIDIA or AMD GPU accelerator;
 * +    DpcppExecutor specifies that the data should be stored and the
 *      operations executed on an hardware supporting DPC++;
 * +    ReferenceExecutor executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 *
 * The following code snippet demonstrates the simplest possible use of the
 * Ginkgo library:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto A = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", omp);
 * ```
 *
 * First, we create a OMP executor, which will be used in the next line to
 * specify where we want the data for the matrix A to be stored.
 * The second line will read a matrix from the matrix market file 'A.mtx',
 * and store the data on the CPU in CSR format (gko::matrix::Csr is a
 * Ginkgo matrix class which stores its data in CSR format).
 * At this point, matrix A is bound to the CPU, and any routines called on it
 * will be performed on the CPU. This approach is usually desired in sparse
 * linear algebra, as the cost of individual operations is several orders of
 * magnitude lower than the cost of copying the matrix to the GPU.
 *
 * If matrix A is going to be reused multiple times, it could be beneficial to
 * copy it over to the accelerator, and perform the operations there, as
 * demonstrated by the next code snippet:
 *
 * ```cpp
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::copy_to<gko::matrix::Csr<float>>(A.get(), cuda);
 * ```
 *
 * The first line of the snippet creates a new CUDA executor. Since there may be
 * multiple NVIDIA GPUs present on the system, the first parameter instructs the
 * library to use the first device (i.e. the one with device ID zero, as in
 * cudaSetDevice() routine from the CUDA runtime API). In addition, since GPUs
 * are not stand-alone processors, it is required to pass a "master" OmpExecutor
 * which will be used to schedule the requested CUDA kernels on the accelerator.
 *
 * The second command creates a copy of the matrix A on the GPU. Notice the use
 * of the get() method. As Ginkgo aims to provide automatic memory
 * management of its objects, the result of calling gko::read_from_mtx()
 * is a smart pointer (std::unique_ptr) to the created object. On the other
 * hand, as the library will not hold a reference to A once the copy is
 * completed, the input parameter for gko::copy_to() is a plain pointer.
 * Thus, the get() method is used to convert from a std::unique_ptr to a
 * plain pointer, as expected by gko::copy_to().
 *
 * As a side note, the gko::copy_to routine is far more powerful than just
 * copying data between different devices. It can also be used to convert data
 * between different formats. For example, if the above code used
 * gko::matrix::Ell as the template parameter, dA would be stored on the GPU,
 * in ELLPACK format.
 *
 * Finally, if all the processing of the matrix is supposed to be done on the
 * GPU, and a CPU copy of the matrix is not required, we could have read the
 * matrix to the GPU directly:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", cuda);
 * ```
 * Notice that even though reading the matrix directly from a file to the
 * accelerator is not supported, the library is designed to abstract away the
 * intermediate step of reading the matrix to the CPU memory. This is a general
 * design approach taken by the library: in case an operation is not supported
 * by the device, the data will be copied to the CPU, the operation performed
 * there, and finally the results copied back to the device.
 * This approach makes using the library more concise, as explicit copies are
 * not required by the user. Nevertheless, this feature should be taken into
 * account when considering performance implications of using such operations.
 *
 * @ingroup Executor
 */
class Executor : public log::EnableLogging<Executor> {
    template <typename T>
    friend class detail::ExecutorBase;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_EXECUTOR_FRIEND);
    friend class ReferenceExecutor;

public:
    virtual ~Executor() = default;

    Executor() = default;
    Executor(Executor &) = delete;
    Executor(Executor &&) = default;
    Executor &operator=(Executor &) = delete;
    Executor &operator=(Executor &&) = default;

    /**
     * Runs the specified Operation using this Executor.
     *
     * @param op  the operation to run
     */
    virtual void run(const Operation &op) const = 0;

    /**
     * Runs one of the passed in functors, depending on the Executor type.
     *
     * @tparam ClosureOmp  type of op_omp
     * @tparam ClosureCuda  type of op_cuda
     * @tparam ClosureHip  type of op_hip
     *
     * @param op_omp  functor to run in case of a OmpExecutor or
     *                ReferenceExecutor
     * @param op_cuda  functor to run in case of a CudaExecutor
     * @param op_hip  functor to run in case of a HipExecutor
     */
    template <typename ClosureOmp, typename ClosureCuda, typename ClosureHip,
              typename ClosureDpcpp>
    void run(const ClosureOmp &op_omp, const ClosureCuda &op_cuda,
             const ClosureHip &op_hip, const ClosureDpcpp &op_dpcpp) const
    {
        LambdaOperation<ClosureOmp, ClosureCuda, ClosureHip, ClosureDpcpp> op(
            op_omp, op_cuda, op_hip, op_dpcpp);
        this->run(op);
    }

    /**
     * Allocates memory in this Executor.
     *
     * @tparam T  datatype to allocate
     *
     * @param num_elems  number of elements of type T to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return pointer to allocated memory
     */
    template <typename T>
    T *alloc(size_type num_elems) const
    {
        this->template log<log::Logger::allocation_started>(
            this, num_elems * sizeof(T));
        T *allocated = static_cast<T *>(this->raw_alloc(num_elems * sizeof(T)));
        this->template log<log::Logger::allocation_completed>(
            this, num_elems * sizeof(T), reinterpret_cast<uintptr>(allocated));
        return allocated;
    }

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    void free(void *ptr) const noexcept
    {
        this->template log<log::Logger::free_started>(
            this, reinterpret_cast<uintptr>(ptr));
        this->raw_free(ptr);
        this->template log<log::Logger::free_completed>(
            this, reinterpret_cast<uintptr>(ptr));
    }

    /**
     * Copies data from another Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param src_exec  Executor from which the memory will be copied
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy_from(const Executor *src_exec, size_type num_elems,
                   const T *src_ptr, T *dest_ptr) const
    {
        this->template log<log::Logger::copy_started>(
            src_exec, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
        try {
            this->raw_copy_from(src_exec, num_elems * sizeof(T), src_ptr,
                                dest_ptr);
        } catch (NotSupported &) {
            // Unoptimized copy. Try to go through the masters.
            auto src_master = src_exec->get_master().get();
            if (num_elems > 0 && src_master != src_exec) {
                auto *master_ptr = src_exec->get_master()->alloc<T>(num_elems);
                src_master->copy_from<T>(src_exec, num_elems, src_ptr,
                                         master_ptr);
                this->copy_from<T>(src_master, num_elems, master_ptr, dest_ptr);
                src_master->free(master_ptr);
            }
        }
        this->template log<log::Logger::copy_completed>(
            src_exec, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
    }

    /**
     * Copies data within this Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy(size_type num_elems, const T *src_ptr, T *dest_ptr) const
    {
        this->copy_from(this, num_elems, src_ptr, dest_ptr);
    }

    /**
     * Retrieves a single element at the given location from executor memory.
     *
     * @tparam T  datatype to copy
     *
     * @param ptr  the pointer to the element to be copied
     *
     * @return the value stored at ptr
     */
    template <typename T>
    T copy_val_to_host(const T *ptr) const
    {
        T out{};
        this->get_master()->copy_from(this, 1, ptr, &out);
        return out;
    }

    /**
     * Returns the master OmpExecutor of this Executor.
     * @return the master OmpExecutor of this Executor.
     */
    virtual std::shared_ptr<Executor> get_master() noexcept = 0;

    /**
     * @copydoc get_master
     */
    virtual std::shared_ptr<const Executor> get_master() const noexcept = 0;

    /**
     * Synchronize the operations launched on the executor with its master.
     */
    virtual void synchronize() const = 0;

    /**
     * Verifies whether the executors share the same memory.
     *
     * @param other  the other Executor to compare against
     *
     * @return whether the executors this and other share the same memory.
     */
    bool memory_accessible(const std::shared_ptr<const Executor> &other) const
    {
        return this->verify_memory_from(other.get());
    }

protected:
    /**
     * Allocates raw memory in this Executor.
     *
     * @param size  number of bytes to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return raw pointer to allocated memory
     */
    virtual void *raw_alloc(size_type size) const = 0;

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void raw_free(void *ptr) const noexcept = 0;

    /**
     * Copies raw data from another Executor.
     *
     * @param src_exec  Executor from which the memory will be copied
     * @param n_bytes  number of bytes to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory where the data
     *                  will be copied to
     */
    virtual void raw_copy_from(const Executor *src_exec, size_type n_bytes,
                               const void *src_ptr, void *dest_ptr) const = 0;

/**
 * @internal
 * Declares a raw_copy_to() overload for a specified Executor subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement raw_copy_from().
 *
 * @param _exec_type  the Executor subclass
 */
#define GKO_ENABLE_RAW_COPY_TO(_exec_type, ...)                              \
    virtual void raw_copy_to(const _exec_type *dest_exec, size_type n_bytes, \
                             const void *src_ptr, void *dest_ptr) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_ENABLE_RAW_COPY_TO);

#undef GKO_ENABLE_RAW_COPY_TO

    /**
     * Verify the memory from another Executor.
     *
     * @param src_exec  Executor from which to verify the memory.
     *
     * @return whether this executor and src_exec share the same memory.
     */
    virtual bool verify_memory_from(const Executor *src_exec) const = 0;

/**
 * @internal
 * Declares a verify_memory_to() overload for a specified Executor subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement verify_memory_from().
 *
 * @param _exec_type  the Executor subclass
 */
#define GKO_ENABLE_VERIFY_MEMORY_TO(_exec_type, ...) \
    virtual bool verify_memory_to(const _exec_type *dest_exec) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_ENABLE_VERIFY_MEMORY_TO);

    GKO_ENABLE_VERIFY_MEMORY_TO(ReferenceExecutor, ref);

#undef GKO_ENABLE_VERIFY_MEMORY_TO

private:
    /**
     * The LambdaOperation class wraps three functor objects into an
     * Operation.
     *
     * The first object is called by the OmpExecutor, the second one by the
     * CudaExecutor and the last one by the HipExecutor. When run on the
     * ReferenceExecutor, the implementation will launch the CPU reference
     * version.
     *
     * @tparam ClosureOmp  the type of the first functor
     * @tparam ClosureCuda  the type of the second functor
     * @tparam ClosureHip  the type of the third functor
     * @tparam ClosureDpcpp  the type of the fourth functor
     */
    template <typename ClosureOmp, typename ClosureCuda, typename ClosureHip,
              typename ClosureDpcpp>
    class LambdaOperation : public Operation {
    public:
        /**
         * Creates an LambdaOperation object from two functors.
         *
         * @param op_omp  a functor object which will be called by OmpExecutor
         *                and ReferenceExecutor
         * @param op_cuda  a functor object which will be called by CudaExecutor
         * @param op_hip  a functor object which will be called by HipExecutor
         * @param op_dpcpp  a functor object which will be called by
         * DpcppExecutor
         */
        LambdaOperation(const ClosureOmp &op_omp, const ClosureCuda &op_cuda,
                        const ClosureHip &op_hip, const ClosureDpcpp &op_dpcpp)
            : op_omp_(op_omp),
              op_cuda_(op_cuda),
              op_hip_(op_hip),
              op_dpcpp_(op_dpcpp)
        {}

        void run(std::shared_ptr<const OmpExecutor>) const override
        {
            op_omp_();
        }

        void run(std::shared_ptr<const CudaExecutor>) const override
        {
            op_cuda_();
        }

        void run(std::shared_ptr<const HipExecutor>) const override
        {
            op_hip_();
        }

        void run(std::shared_ptr<const DpcppExecutor>) const override
        {
            op_dpcpp_();
        }

    private:
        ClosureOmp op_omp_;
        ClosureCuda op_cuda_;
        ClosureHip op_hip_;
        ClosureDpcpp op_dpcpp_;
    };
};


/**
 * This is a deleter that uses an executor's `free` method to deallocate the
 * data.
 *
 * @tparam T  the type of object being deleted
 *
 * @ingroup Executor
 */
template <typename T>
class executor_deleter {
public:
    using pointer = T *;

    /**
     * Creates a new deleter.
     *
     * @param exec  the executor used to free the data
     */
    explicit executor_deleter(std::shared_ptr<const Executor> exec)
        : exec_{exec}
    {}

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const
    {
        if (exec_) {
            exec_->free(ptr);
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
};

// a specialization for arrays
template <typename T>
class executor_deleter<T[]> {
public:
    using pointer = T[];

    explicit executor_deleter(std::shared_ptr<const Executor> exec)
        : exec_{exec}
    {}

    void operator()(pointer ptr) const
    {
        if (exec_) {
            exec_->free(ptr);
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
};


namespace detail {


template <typename ConcreteExecutor>
class ExecutorBase : public Executor {
    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_EXECUTOR_FRIEND);
    friend class ReferenceExecutor;

public:
    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(self()->shared_from_this());
        this->template log<log::Logger::operation_completed>(this, &op);
    }

protected:
    void raw_copy_from(const Executor *src_exec, size_type n_bytes,
                       const void *src_ptr, void *dest_ptr) const override
    {
        src_exec->raw_copy_to(self(), n_bytes, src_ptr, dest_ptr);
    }

    virtual bool verify_memory_from(const Executor *src_exec) const override
    {
        return src_exec->verify_memory_to(self());
    }

private:
    ConcreteExecutor *self() noexcept
    {
        return static_cast<ConcreteExecutor *>(this);
    }

    const ConcreteExecutor *self() const noexcept
    {
        return static_cast<const ConcreteExecutor *>(this);
    }
};

#undef GKO_DECLARE_EXECUTOR_FRIEND

/**
 * Controls whether the DeviceReset function should be called thanks to a
 * boolean. Note that in any case, `DeviceReset` is called only after destroying
 * the last Ginkgo executor. Therefore, it is sufficient to set this flag to the
 * last living executor in Ginkgo. Setting this flag to an executor which is not
 * destroyed last has no effect.
 */
class EnableDeviceReset {
public:
    /**
     * Set the device reset capability.
     *
     * @param device_reset  whether to allow a device reset or not
     */
    void set_device_reset(bool device_reset) { device_reset_ = device_reset; }

    /**
     * Returns the current status of the device reset boolean for this executor.
     *
     * @return the current status of the device reset boolean for this executor.
     */
    bool get_device_reset() { return device_reset_; }

protected:
    /**
     * Instantiate an EnableDeviceReset class
     *
     * @param device_reset  the starting device_reset status. Defaults to false.
     */
    EnableDeviceReset(bool device_reset = false) : device_reset_{device_reset}
    {}

private:
    bool device_reset_{};
};


}  // namespace detail


#define GKO_OVERRIDE_RAW_COPY_TO(_executor_type, ...)                    \
    void raw_copy_to(const _executor_type *dest_exec, size_type n_bytes, \
                     const void *src_ptr, void *dest_ptr) const override


#define GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(dest_, bool_)                     \
    virtual bool verify_memory_to(const dest_ *other) const override         \
    {                                                                        \
        return bool_;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * This is the Executor subclass which represents the OpenMP device
 * (typically CPU).
 *
 * @ingroup exec_omp
 * @ingroup Executor
 */
class OmpExecutor : public detail::ExecutorBase<OmpExecutor>,
                    public std::enable_shared_from_this<OmpExecutor> {
    friend class detail::ExecutorBase<OmpExecutor>;

public:
    /**
     * Creates a new OmpExecutor.
     */
    static std::shared_ptr<OmpExecutor> create()
    {
        return std::shared_ptr<OmpExecutor>(new OmpExecutor());
    }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

protected:
    OmpExecutor() = default;

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    bool verify_memory_to(const DpcppExecutor *dest_exec) const override;
};


namespace kernels {
namespace omp {
using DefaultExecutor = OmpExecutor;
}  // namespace omp
}  // namespace kernels


/**
 * This is a specialization of the OmpExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
 *
 * @ingroup exec_ref
 * @ingroup Executor
 */
class ReferenceExecutor : public OmpExecutor {
public:
    static std::shared_ptr<ReferenceExecutor> create()
    {
        return std::shared_ptr<ReferenceExecutor>(new ReferenceExecutor());
    }

    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(std::static_pointer_cast<const ReferenceExecutor>(
            this->shared_from_this()));
        this->template log<log::Logger::operation_completed>(this, &op);
    }

protected:
    ReferenceExecutor() = default;

    bool verify_memory_from(const Executor *src_exec) const override
    {
        return src_exec->verify_memory_to(this);
    }

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);
};


namespace kernels {
namespace reference {
using DefaultExecutor = ReferenceExecutor;
}  // namespace reference
}  // namespace kernels


/**
 * This is the Executor subclass which represents the CUDA device.
 *
 * @ingroup exec_cuda
 * @ingroup Executor
 */
class CudaExecutor : public detail::ExecutorBase<CudaExecutor>,
                     public std::enable_shared_from_this<CudaExecutor>,
                     public detail::EnableDeviceReset {
    friend class detail::ExecutorBase<CudaExecutor>;

public:
    /**
     * Creates a new CudaExecutor.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        bool device_reset = false);

    ~CudaExecutor() { decrease_num_execs(this->device_id_); }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    void run(const Operation &op) const override;

    /**
     * Get the CUDA device id of the device associated to this executor.
     */
    int get_device_id() const noexcept { return device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept { return num_warps_per_sm_; }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept { return num_multiprocessor_; }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return num_multiprocessor_ * num_warps_per_sm_;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept { return warp_size_; }

    /**
     * Get the major verion of compute capability.
     */
    int get_major_version() const noexcept { return major_; }

    /**
     * Get the minor verion of compute capability.
     */
    int get_minor_version() const noexcept { return minor_; }

    /**
     * Get the cublas handle for this executor
     *
     * @return  the cublas handle (cublasContext*) for this executor
     */
    cublasContext *get_cublas_handle() const { return cublas_handle_.get(); }

    /**
     * Get the cusparse handle for this executor
     *
     * @return the cusparse handle (cusparseContext*) for this executor
     */
    cusparseContext *get_cusparse_handle() const
    {
        return cusparse_handle_.get();
    }

protected:
    void set_gpu_property();

    void init_handles();

    CudaExecutor(int device_id, std::shared_ptr<Executor> master,
                 bool device_reset = false)
        : EnableDeviceReset{device_reset},
          device_id_(device_id),
          master_(master),
          num_warps_per_sm_(0),
          num_multiprocessor_(0),
          major_(0),
          minor_(0),
          warp_size_(0)
    {
        assert(device_id < max_devices && device_id >= 0);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
    }

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    bool verify_memory_to(const HipExecutor *dest_exec) const override;

    bool verify_memory_to(const CudaExecutor *dest_exec) const override;

    static void increase_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]++;
    }

    static void decrease_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]--;
    }

    static unsigned get_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        return num_execs[device_id];
    }

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    int num_warps_per_sm_;
    int num_multiprocessor_;
    int major_;
    int minor_;
    int warp_size_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<cublasContext> cublas_handle_;
    handle_manager<cusparseContext> cusparse_handle_;

    static constexpr int max_devices = 64;
    static unsigned num_execs[max_devices];
    static std::mutex mutex[max_devices];
};


namespace kernels {
namespace cuda {
using DefaultExecutor = CudaExecutor;
}  // namespace cuda
}  // namespace kernels


/**
 * This is the Executor subclass which represents the HIP enhanced device.
 *
 * @ingroup exec_hip
 * @ingroup Executor
 */
class HipExecutor : public detail::ExecutorBase<HipExecutor>,
                    public std::enable_shared_from_this<HipExecutor>,
                    public detail::EnableDeviceReset {
    friend class detail::ExecutorBase<HipExecutor>;

public:
    /**
     * Creates a new HipExecutor.
     *
     * @param device_id  the HIP device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels
     */
    static std::shared_ptr<HipExecutor> create(int device_id,
                                               std::shared_ptr<Executor> master,
                                               bool device_reset = false);

    ~HipExecutor() { decrease_num_execs(this->device_id_); }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    void run(const Operation &op) const override;

    /**
     * Get the HIP device id of the device associated to this executor.
     */
    int get_device_id() const noexcept { return device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept { return num_warps_per_sm_; }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept { return num_multiprocessor_; }

    /**
     * Get the major verion of compute capability.
     */
    int get_major_version() const noexcept { return major_; }

    /**
     * Get the minor verion of compute capability.
     */
    int get_minor_version() const noexcept { return minor_; }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return num_multiprocessor_ * num_warps_per_sm_;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept { return warp_size_; }

    /**
     * Get the hipblas handle for this executor
     *
     * @return  the hipblas handle (hipblasContext*) for this executor
     */
    hipblasContext *get_hipblas_handle() const { return hipblas_handle_.get(); }

    /**
     * Get the hipsparse handle for this executor
     *
     * @return the hipsparse handle (hipsparseContext*) for this executor
     */
    hipsparseContext *get_hipsparse_handle() const
    {
        return hipsparse_handle_.get();
    }

protected:
    void set_gpu_property();

    void init_handles();

    HipExecutor(int device_id, std::shared_ptr<Executor> master,
                bool device_reset = false)
        : EnableDeviceReset{device_reset},
          device_id_(device_id),
          master_(master),
          num_multiprocessor_(0),
          num_warps_per_sm_(0),
          major_(0),
          minor_(0),
          warp_size_(0)
    {
        assert(device_id < max_devices);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
    }

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    bool verify_memory_to(const CudaExecutor *dest_exec) const override;

    bool verify_memory_to(const HipExecutor *dest_exec) const override;

    static void increase_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]++;
    }

    static void decrease_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]--;
    }

    static int get_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        return num_execs[device_id];
    }

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    int num_multiprocessor_;
    int num_warps_per_sm_;
    int major_;
    int minor_;
    int warp_size_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<hipblasContext> hipblas_handle_;
    handle_manager<hipsparseContext> hipsparse_handle_;

    static constexpr int max_devices = 64;
    static int num_execs[max_devices];
    static std::mutex mutex[max_devices];
};


namespace kernels {
namespace hip {
using DefaultExecutor = HipExecutor;
}  // namespace hip
}  // namespace kernels


/**
 * This is the Executor subclass which represents a DPC++ enhanced device.
 *
 * @ingroup exec_dpcpp
 * @ingroup Executor
 */
class DpcppExecutor : public detail::ExecutorBase<DpcppExecutor>,
                      public std::enable_shared_from_this<DpcppExecutor> {
    friend class detail::ExecutorBase<DpcppExecutor>;

public:
    /**
     * Creates a new DpcppExecutor.
     *
     * @param device_id  the DPCPP device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels
     * @param device_type  a string representing the type of device to consider
     *                     (accelerator, cpu, gpu or all).
     */
    static std::shared_ptr<DpcppExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        std::string device_type = "all");

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    void run(const Operation &op) const override;

    /**
     * Get the DPCPP device id of the device associated to this executor.
     *
     * @return the DPCPP device id of the device associated to this executor
     */
    int get_device_id() const noexcept { return device_id_; }

    ::cl::sycl::queue *get_queue() const { return queue_.get(); }

    /**
     * Get the number of devices present on the system.
     *
     * @param device_type  a string representing the device type
     *
     * @return the number of devices present on the system
     */
    static int get_num_devices(std::string device_type);

    /**
     * Get the available subgroup sizes for this device.
     *
     * @return the available subgroup sizes for this device
     */
    const std::vector<size_type> &get_subgroup_sizes() const noexcept
    {
        return subgroup_sizes_;
    }

    /**
     * Get the number of Computing Units of this executor.
     *
     * @return the number of Computing Units of this executor
     */
    size_type get_num_computing_units() const noexcept
    {
        return num_computing_units_;
    }

    /**
     * Get the maximum work item sizes.
     *
     * @return the maximum work item sizes
     */
    const std::vector<size_type> &get_max_workitem_sizes() const noexcept
    {
        return max_workitem_sizes_;
    }

    /**
     * Get the maximum workgroup size.
     *
     * @return the maximum workgroup size
     */
    size_type get_max_workgroup_size() const noexcept
    {
        return max_workgroup_size_;
    }

    /**
     * Get a string representing the device type.
     *
     * @return a string representing the device type
     */
    std::string get_device_type() const noexcept { return device_type_; }

protected:
    void set_device_property();

    DpcppExecutor(int device_id, std::shared_ptr<Executor> master,
                  std::string device_type = "all")
        : device_id_(device_id), master_(master), device_type_(device_type)
    {
        std::for_each(device_type_.begin(), device_type_.end(),
                      [](char &c) { c = std::tolower(c); });
        this->set_device_property();
    }

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    bool verify_memory_to(const OmpExecutor *dest_exec) const override;

    bool verify_memory_to(const DpcppExecutor *dest_exec) const override;

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    std::string device_type_;
    int num_computing_units_{};
    std::vector<size_type> subgroup_sizes_{};
    std::vector<size_type> max_workitem_sizes_{};
    size_type max_workgroup_size_{};

    template <typename T>
    using queue_manager = std::unique_ptr<T, std::function<void(T *)>>;
    queue_manager<::cl::sycl::queue> queue_;
};


namespace kernels {
namespace dpcpp {
using DefaultExecutor = DpcppExecutor;
}  // namespace dpcpp
}  // namespace kernels


#undef GKO_OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_
