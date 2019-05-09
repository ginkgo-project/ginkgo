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

#ifndef __CUDA_LINOPS__
#define __CUDA_LINOPS__

#include <ginkgo/ginkgo.hpp>


#include "cuda/base/cusparse_bindings.hpp"


#include <cuda_runtime.h>
#include <cusparse.h>
#include <memory>


namespace detail {


class CuspBase : public gko::LinOp {
public:
    void apply_impl(const gko::LinOp *, const gko::LinOp *, const gko::LinOp *,
                    gko::LinOp *) const override
    {}

    cusparseMatDescr_t get_descr() const { return this->descr_.get(); }

    const gko::CudaExecutor *get_gpu_exec() const { return gpu_exec; }

protected:
    CuspBase(std::shared_ptr<const gko::Executor> exec,
             const gko::dim<2> &size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        gpu_exec = dynamic_cast<const gko::CudaExecutor *>(exec.get());
        if (gpu_exec == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_descr();
    }

    CuspBase &operator=(const CuspBase &other)
    {
        this->gpu_exec = other.get_gpu_exec();
        this->initialize_descr();
        return *this;
    }

    void initialize_descr()
    {
        const auto id = this->gpu_exec->get_device_id();
        this->descr_ = handle_manager<cusparseMatDescr>(
            gko::kernels::cuda::cusparse::create_mat_descr(),
            [id](cusparseMatDescr_t descr) {
                int original_device_id{};
                GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id));
                GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(id));
                gko::kernels::cuda::cusparse::destroy(descr);
                cudaSetDevice(original_device_id);
            });
    }

private:
    const gko::CudaExecutor *gpu_exec;
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

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        Csr_->read(data);
        this->set_size(gko::dim<2>{Csr_->get_size()});
    }

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<double>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrmv_mp(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            Csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            Csr_->get_const_values(), Csr_->get_const_row_ptrs(),
            Csr_->get_const_col_idxs(), db, &beta, dx));
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return Csr_->get_num_stored_elements();
    }

    CuspCsrmp(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrmp, CuspBase>(exec, size),
          Csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}


private:
    std::shared_ptr<csr> Csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsr
    : public gko::EnableLinOp<CuspCsr<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsr<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsr>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        Csr_->read(data);
        this->set_size(gko::dim<2>{Csr_->get_size()});
    }

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrmv(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            Csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            Csr_->get_const_values(), Csr_->get_const_row_ptrs(),
            Csr_->get_const_col_idxs(), db, &beta, dx));
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return Csr_->get_num_stored_elements();
    }

    CuspCsr(std::shared_ptr<const gko::Executor> exec,
            const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsr, CuspBase>(exec, size),
          Csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    std::shared_ptr<csr> Csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsrmm
    : public gko::EnableLinOp<CuspCsrmm<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsrmm<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsrmm>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        Csr_->read(data);
        this->set_size(gko::dim<2>{Csr_->get_size()});
    }

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();

        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrmm(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], dense_b->get_size()[1], this->get_size()[1],
            Csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            Csr_->get_const_values(), Csr_->get_const_row_ptrs(),
            Csr_->get_const_col_idxs(), db, dense_b->get_size()[0], &beta, dx,
            dense_x->get_size()[0]));
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return Csr_->get_num_stored_elements();
    }

    CuspCsrmm(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrmm, CuspBase>(exec, size),
          Csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    std::shared_ptr<csr> Csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CuspCsrEx
    : public gko::EnableLinOp<CuspCsrEx<ValueType, IndexType>, CuspBase>,
      public gko::EnableCreateMethod<CuspCsrEx<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CuspCsrEx>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        Csr_->read(data);
        size_t buffer_size;
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        this->set_size(gko::dim<2>{Csr_->get_size()});

        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx_bufferSize(
            this->get_gpu_exec()->get_cusparse_handle(), algmode_, trans_,
            this->get_size()[0], this->get_size()[1],
            Csr_->get_num_stored_elements(), &alpha, CUDA_R_64F,
            this->get_descr(), Csr_->get_const_values(), CUDA_R_64F,
            Csr_->get_const_row_ptrs(), Csr_->get_const_col_idxs(), nullptr,
            CUDA_R_64F, &beta, CUDA_R_64F, nullptr, CUDA_R_64F, CUDA_R_64F,
            &buffer_size));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMalloc(&buffer_, buffer_size));
        set_buffer_ = true;
    }

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx(
            this->get_gpu_exec()->get_cusparse_handle(), algmode_, trans_,
            this->get_size()[0], this->get_size()[1],
            Csr_->get_num_stored_elements(), &alpha, CUDA_R_64F,
            this->get_descr(), Csr_->get_const_values(), CUDA_R_64F,
            Csr_->get_const_row_ptrs(), Csr_->get_const_col_idxs(), db,
            CUDA_R_64F, &beta, CUDA_R_64F, dx, CUDA_R_64F, CUDA_R_64F,
            buffer_));
    }


    void apply_impl(const gko::LinOp *, const gko::LinOp *, const gko::LinOp *,
                    gko::LinOp *) const override
    {}

    gko::size_type get_num_stored_elements() const noexcept
    {
        return Csr_->get_num_stored_elements();
    }

    CuspCsrEx(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspCsrEx, CuspBase>(exec, size),
          Csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE),
          set_buffer_(false)
    {
#ifdef ALLOWMP
        cusparseAlgMode_t algmode_ = CUSPARSE_ALG_MERGE_PATH;
#endif
    }

    ~CuspCsrEx() override
    {
        if (set_buffer_) {
            try {
                GKO_ASSERT_NO_CUDA_ERRORS(cudaFree(buffer_));
            } catch (std::exception &e) {
                std::cout
                    << "Error when unallocating CuspCsrEx temporary buffer"
                    << std::endl;
            }
        }
    }

private:
    std::shared_ptr<csr> Csr_;
    cusparseOperation_t trans_;
    cusparseAlgMode_t algmode_;
    void *buffer_;
    bool set_buffer_;
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

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override
    {
        auto t_Csr = csr::create(this->get_executor(),
                                 std::make_shared<typename csr::classical>());
        t_Csr->read(data);
        this->set_size(gko::dim<2>{t_Csr->get_size()});
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsr2hyb(
            this->get_gpu_exec()->get_cusparse_handle(), this->get_size()[0],
            this->get_size()[1], this->get_descr(), t_Csr->get_const_values(),
            t_Csr->get_const_row_ptrs(), t_Csr->get_const_col_idxs(), hyb_,
            Threshold, Partition));
    }

    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseDhybmv(this->get_gpu_exec()->get_cusparse_handle(), trans_,
                           &alpha, this->get_descr(), hyb_, db, &beta, dx));
    }

    CuspHybrid(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<CuspHybrid, CuspBase>(exec, size),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateHybMat(&hyb_));
    }

    ~CuspHybrid() override
    {
        try {
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyHybMat(hyb_));
        } catch (std::exception &e) {
            std::cout << "Error when unallocating CuspHybrid hyb_ matrix"
                      << std::endl;
        }
    }

private:
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

#endif  // __CUDA_LINOPS__
