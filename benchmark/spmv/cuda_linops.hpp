/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_BENCHMARK_SPMV_CUDA_LINOPS_HPP_
#define GKO_BENCHMARK_SPMV_CUDA_LINOPS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <cuda_runtime.h>
#include <cusparse.h>
#include <memory>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/device_guard.hpp"
#include "cuda/base/pointer_mode_guard.hpp"


namespace detail {


class CuspBase : public gko::LinOp {
public:
    cusparseMatDescr_t get_descr() const { return this->descr_.get(); }

    const gko::CudaExecutor *get_gpu_exec() const { return gpu_exec_.get(); }

protected:
    void apply_impl(const gko::LinOp *, const gko::LinOp *, const gko::LinOp *,
                    gko::LinOp *) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    CuspBase(std::shared_ptr<const gko::Executor> exec,
             const gko::dim<2> &size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        gpu_exec_ = std::dynamic_pointer_cast<const gko::CudaExecutor>(exec);
        if (gpu_exec_ == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_descr();
    }

    CuspBase &operator=(const CuspBase &other)
    {
        gko::LinOp::operator=(other);
        this->gpu_exec_ = other.gpu_exec_;
        this->initialize_descr();
        return *this;
    }

    void initialize_descr()
    {
        const auto id = this->gpu_exec_->get_device_id();
        gko::device_guard g{id};
        this->descr_ = handle_manager<cusparseMatDescr>(
            gko::kernels::cuda::cusparse::create_mat_descr(),
            [id](cusparseMatDescr_t descr) {
                gko::device_guard g{id};
                gko::kernels::cuda::cusparse::destroy(descr);
            });
    }

private:
    std::shared_ptr<const gko::CudaExecutor> gpu_exec_;
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<cusparseMatDescr> descr_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsrmp
    : public gko::EnableLinOp<CuspCsrmp<ValueType, IndexType>, CuspBase>,
      public gko::ReadableFromMatrixData<ValueType, IndexType>,
      public gko::EnableCreateMethod<CuspCsrmp<ValueType, IndexType>> {
    friend class gko::EnableCreateMethod<CuspCsrmp>;
    friend class gko::EnablePolymorphicObject<CuspCsrmp, CuspBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        csr_->read(data);
        this->set_size(gko::dim<2>{csr_->get_size()});
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<double>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        gko::kernels::cuda::cusparse::spmv_mp(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    CuspCsrmp(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrmp, CuspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsr
    : public gko::EnableLinOp<CuspCsr<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsr<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsr>;
    friend class gko::EnablePolymorphicObject<CuspCsr, CuspBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        csr_->read(data);
        this->set_size(gko::dim<2>{csr_->get_size()});
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        gko::kernels::cuda::cusparse::spmv(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    CuspCsr(std::shared_ptr<const gko::Executor> exec,
            const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsr, CuspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsrmm
    : public gko::EnableLinOp<CuspCsrmm<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsrmm<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsrmm>;
    friend class gko::EnablePolymorphicObject<CuspCsrmm, CuspBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        csr_->read(data);
        this->set_size(gko::dim<2>{csr_->get_size()});
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        gko::kernels::cuda::cusparse::spmm(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], dense_b->get_size()[1], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            dense_b->get_size()[0], &scalars.get_const_data()[1], dx,
            dense_x->get_size()[0]);
    }

    CuspCsrmm(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrmm, CuspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsrEx
    : public gko::EnableLinOp<CuspCsrEx<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsrEx<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsrEx>;
    friend class gko::EnablePolymorphicObject<CuspCsrEx, CuspBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        csr_->read(data);
        this->set_size(gko::dim<2>{csr_->get_size()});
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    ~CuspCsrEx() override
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        if (set_buffer_) {
            try {
                GKO_ASSERT_NO_CUDA_ERRORS(cudaFree(buffer_));
            } catch (const std::exception &e) {
                std::cerr
                    << "Error when unallocating CuspCsrEx temporary buffer: "
                    << e.what() << std::endl;
            }
        }
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        ValueType alpha = gko::one<ValueType>();
        ValueType beta = gko::zero<ValueType>();
        gko::size_type buffer_size = 0;

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        auto handle = this->get_gpu_exec()->get_cusparse_handle();
        // This function seems to require the pointer mode to be set to HOST.
        // Ginkgo use pointer mode DEVICE by default, so we change this
        // temporarily.
        gko::cusparse_pointer_mode_guard pm_guard(handle);
        gko::kernels::cuda::cusparse::spmv_buffersize<ValueType, IndexType>(
            handle, algmode_, trans_, this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, &beta, dx, &buffer_size);
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMalloc(&buffer_, buffer_size));
        set_buffer_ = true;

        gko::kernels::cuda::cusparse::spmv<ValueType, IndexType>(
            handle, algmode_, trans_, this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, &beta, dx, buffer_);

        // Exiting the scope sets the pointer mode back to the default
        // DEVICE for Ginkgo
    }


    CuspCsrEx(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrEx, CuspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE),
          set_buffer_(false)
    {
#ifdef ALLOWMP
        algmode_ = CUSPARSE_ALG_MERGE_PATH;
#endif
    }

private:
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
    cusparseAlgMode_t algmode_;
    mutable void *buffer_;
    mutable bool set_buffer_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          cusparseHybPartition_t Partition = CUSPARSE_HYB_PARTITION_AUTO,
          int Threshold = 0>
class CuspHybrid
    : public gko::EnableLinOp<
          CuspHybrid<ValueType, IndexType, Partition, Threshold>, CuspBase>,
      public gko::EnableCreateMethod<
          CuspHybrid<ValueType, IndexType, Partition, Threshold>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspHybrid>;
    friend class gko::EnablePolymorphicObject<CuspHybrid, CuspBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        auto t_csr = csr::create(this->get_executor(),
                                 std::make_shared<typename csr::classical>());
        t_csr->read(data);
        this->set_size(gko::dim<2>{t_csr->get_size()});

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        gko::kernels::cuda::cusparse::csr2hyb(
            this->get_gpu_exec()->get_cusparse_handle(), this->get_size()[0],
            this->get_size()[1], this->get_descr(), t_csr->get_const_values(),
            t_csr->get_const_row_ptrs(), t_csr->get_const_col_idxs(), hyb_,
            Threshold, Partition);
    }

    ~CuspHybrid() override
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        try {
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyHybMat(hyb_));
        } catch (const std::exception &e) {
            std::cerr << "Error when unallocating CuspHybrid hyb_ matrix: "
                      << e.what() << std::endl;
        }
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        gko::kernels::cuda::cusparse::spmv(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            &scalars.get_const_data()[0], this->get_descr(), hyb_, db,
            &scalars.get_const_data()[1], dx);
    }

    CuspHybrid(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspHybrid, CuspBase>(exec, size),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateHybMat(&hyb_));
    }

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    cusparseOperation_t trans_;
    cusparseHybMat_t hyb_;
};


}  // namespace detail


// Some shortcuts
using cusp_csr = detail::CuspCsr<>;
using cusp_csrex = detail::CuspCsrEx<>;
using cusp_csrmp = detail::CuspCsrmp<>;
using cusp_csrmm = detail::CuspCsrmm<>;


using cusp_coo =
    detail::CuspHybrid<double, gko::int32, CUSPARSE_HYB_PARTITION_USER, 0>;
using cusp_ell =
    detail::CuspHybrid<double, gko::int32, CUSPARSE_HYB_PARTITION_MAX, 0>;
using cusp_hybrid = detail::CuspHybrid<>;

#endif  // GKO_BENCHMARK_SPMV_CUDA_LINOPS_HPP_
