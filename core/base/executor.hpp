/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_EXECUTOR_HPP_
#define GKO_CORE_EXECUTOR_HPP_


#include <memory>
#include <tuple>
#include <type_traits>


#include "core/base/types.hpp"
#include "core/log/logger.hpp"


struct cublasContext;

struct cusparseContext;


namespace gko {


#define FORWARD_DECLARE(_type, _unused) class _type

GKO_ENABLE_FOR_ALL_EXECUTORS(FORWARD_DECLARE);

#undef FORWARD_DECLARE


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
 *           << *gko::ReferenceExecutor::create() << std::endl;
 * ```
 *
 * which produces the expected output:
 *
 * ```
 * OMP
 * CUDA(0)
 * Reference CPU
 * ```
 *
 * One might feel that this code is too complicated for such a simple task.
 * Luckily, there is an overload of the Executor::run() method, which is
 * designed to facilitate writing simple operations like this one. The method
 * takes two closures as input: one which is run for OMP, and the other one for
 * CUDA executors. Using this method, there is no need to implement an Operation
 * subclass:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     exec.run(
 *         [&]() { os << "OMP"; },  // OMP closure
 *         [&]() { os << "CUDA("    // CUDA closure
 *                    << static_cast<gko::CudaExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *     return os;
 * }
 * ```
 *
 * Using this approach, however, it is impossible to distinguish between
 * a OmpExecutor and ReferenceExecutor, as both of them call the OMP closure.
 */
class Operation {
public:
#define DECLARE_RUN_OVERLOAD(_type, _unused) \
    virtual void run(std::shared_ptr<const _type>) const

    GKO_ENABLE_FOR_ALL_EXECUTORS(DECLARE_RUN_OVERLOAD);

#undef DECLARE_RUN_OVERLOAD

    // ReferenceExecutor overload can be defaulted to OmpExecutor's
    virtual void run(std::shared_ptr<const ReferenceExecutor> executor) const;
};


namespace detail {


template <int K, int... Ns, typename F, typename Exec, typename Tuple>
typename std::enable_if<(K == 0)>::type call_impl(
    F f, std::shared_ptr<const Exec> &exec, Tuple &data)
{
    f(exec, std::get<Ns>(data)...);
}

template <int K, int... Ns, typename F, typename Exec, typename Tuple>
typename std::enable_if<(K > 0)>::type call_impl(
    F f, std::shared_ptr<const Exec> &exec, Tuple &data)
{
    call_impl<K - 1, K - 1, Ns...>(f, exec, data);
}

template <typename F, typename Exec, typename... Args>
void call(F f, std::shared_ptr<const Exec> &exec, std::tuple<Args...> &data)
{
    call_impl<sizeof...(Args)>(f, exec, data);
}


}  // namespace detail


/**
 * Binds a set of device-specific kernels to an Operation.
 *
 * It also defines a helper function which creates the associated operation.
 * Any input arguments passed to the helper function are forwarded to the
 * kernel when the operation is executed.
 *
 * The kernels used to bind the operation are searched in `kernels::DEV_TYPE`
 * namespace, where `DEV_TYPE` is replaced by `omp`, `cuda` and `reference`.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * // define the omp, cuda and reference kernels which will be bound to the
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
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op_operation(5); // x = 5
 *
 *     omp->run(op);  // run omp kernel
 *     cuda->run(op);  // run cuda kernel
 *     ref->run(op);  // run reference kernel
 * }
 * ```
 */
#define GKO_REGISTER_OPERATION(_name, _kernel)                                 \
    template <typename... Args>                                                \
    class _name##_operation : public Operation {                               \
    public:                                                                    \
        _name##_operation(Args &&... args) : data(std::forward<Args>(args)...) \
        {}                                                                     \
                                                                               \
        void run(std::shared_ptr<const OmpExecutor> exec) const override       \
        {                                                                      \
            ::gko::detail::call(::gko::kernels::omp::_kernel, exec, data);     \
        }                                                                      \
                                                                               \
        void run(std::shared_ptr<const CudaExecutor> exec) const override      \
        {                                                                      \
            ::gko::detail::call(::gko::kernels::cuda::_kernel, exec, data);    \
        }                                                                      \
                                                                               \
        void run(std::shared_ptr<const ReferenceExecutor> exec) const override \
        {                                                                      \
            ::gko::detail::call(::gko::kernels::reference::_kernel, exec,      \
                                data);                                         \
        }                                                                      \
                                                                               \
    private:                                                                   \
        mutable std::tuple<Args &&...> data;                                   \
    };                                                                         \
                                                                               \
    template <typename... Args>                                                \
    static _name##_operation<Args...> make_##_name##_operation(                \
        Args &&... args)                                                       \
    {                                                                          \
        return _name##_operation<Args...>(std::forward<Args>(args)...);        \
    }


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
 */
class Executor : public log::EnableLogging<Executor> {
    template <typename T>
    friend class detail::ExecutorBase;

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
     *
     * @param op_omp  functor to run in case of a OmpExecutor or
     *                ReferenceExecutor
     * @param op_cuda  functor to run in case of a CudaExecutor
     */
    template <typename ClosureOmp, typename ClosureCuda>
    void run(const ClosureOmp &op_omp, const ClosureCuda &op_cuda) const
    {
        LambdaOperation<ClosureOmp, ClosureCuda> op(op_omp, op_cuda);
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
        this->raw_copy_from(src_exec, num_elems * sizeof(T), src_ptr, dest_ptr);
        this->template log<log::Logger::copy_completed>(
            src_exec, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
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
#define ENABLE_RAW_COPY_TO(_exec_type, _unused)                              \
    virtual void raw_copy_to(const _exec_type *dest_exec, size_type n_bytes, \
                             const void *src_ptr, void *dest_ptr) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(ENABLE_RAW_COPY_TO);

#undef ENABLE_RAW_COPY_TO

private:
    /**
     * The LambdaOperation class wraps two functor objects into an Operation.
     *
     * The first object is called by the OmpExecutor, while the other one by the
     * CudaExecutor. When run on the ReferenceExecutor, the implementation will
     * launch the CPU reference version.
     *
     * @tparam ClosureOmp  the type of the first functor
     * @tparam ClosureCuda  the type of the second functor
     */
    template <typename ClosureOmp, typename ClosureCuda>
    class LambdaOperation : public Operation {
    public:
        /**
         * Creates an LambdaOperation object from two functors.
         *
         * @param op_omp  a functor object which will be called by OmpExecutor
         *                and ReferenceExecutor
         * @param op_cuda  a functor object which will be called by CudaExecutor
         */
        LambdaOperation(const ClosureOmp &op_omp, const ClosureCuda &op_cuda)
            : op_omp_(op_omp), op_cuda_(op_cuda)
        {}

        void run(std::shared_ptr<const OmpExecutor>) const override
        {
            op_omp_();
        }

        void run(std::shared_ptr<const CudaExecutor>) const override
        {
            op_cuda_();
        }

    private:
        ClosureOmp op_omp_;
        ClosureCuda op_cuda_;
    };
};


/**
 * This is a deleter that uses an executor's `free` method to deallocate the
 * data.
 *
 * @tparam T  the type of object being deleted
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


}  // namespace detail


#define OVERRIDE_RAW_COPY_TO(_executor_type, _unused)                    \
    void raw_copy_to(const _executor_type *dest_exec, size_type n_bytes, \
                     const void *src_ptr, void *dest_ptr) const override


/**
 * This is the Executor subclass which represents the OpenMP device
 * (typically CPU).
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

    GKO_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);
};


namespace kernels {
namespace omp {
using DefaultExecutor = OmpExecutor;
}  // namespace omp
}  // namespace kernels


/**
 * This is a specialization of the OmpExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
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
};


namespace kernels {
namespace reference {
using DefaultExecutor = ReferenceExecutor;
}  // namespace reference
}  // namespace kernels


/**
 * This is the Executor subclass which represents the CUDA device.
 */
class CudaExecutor : public detail::ExecutorBase<CudaExecutor>,
                     public std::enable_shared_from_this<CudaExecutor> {
    friend class ExecutorBase<CudaExecutor>;

public:
    /**
     * Creates a new CudaExecutor.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<Executor> master)
    {
        return std::shared_ptr<CudaExecutor>(
            new CudaExecutor(device_id, std::move(master)));
    }

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
     * Get the number of cores per SM of this executor.
     */
    int get_num_cores_per_sm() const noexcept { return num_cores_per_sm_; }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept { return num_multiprocessor_; }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        constexpr uint32 warp_size = 32;
        auto warps_per_sm = num_cores_per_sm_ / warp_size;
        return num_multiprocessor_ * warps_per_sm;
    }

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

    CudaExecutor(int device_id, std::shared_ptr<Executor> master)
        : device_id_(device_id),
          master_(master),
          num_cores_per_sm_(0),
          num_multiprocessor_(0),
          major_(0),
          minor_(0)
    {
        this->set_gpu_property();
        this->init_handles();
    }

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    int num_cores_per_sm_;
    int num_multiprocessor_;
    int major_;
    int minor_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<cublasContext> cublas_handle_;
    handle_manager<cusparseContext> cusparse_handle_;
};


namespace kernels {
namespace cuda {
using DefaultExecutor = CudaExecutor;
}  // namespace cuda
}  // namespace kernels


#undef OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_CORE_EXECUTOR_HPP_
