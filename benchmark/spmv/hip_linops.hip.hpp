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

#ifndef GKO_BENCHMARK_SPMV_HIP_LINOPS_HPP_
#define GKO_BENCHMARK_SPMV_HIP_LINOPS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <hipsparse.h>
#include <memory>


#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/device_guard.hip.hpp"


namespace detail {


class HipspBase : public gko::LinOp {
public:
    hipsparseMatDescr * get_descr() const { return this->descr_.get(); }

    const gko::HipExecutor *get_gpu_exec() const { return gpu_exec_.get(); }

protected:
    void apply_impl(const gko::LinOp *, const gko::LinOp *, const gko::LinOp *,
                    gko::LinOp *) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    HipspBase(std::shared_ptr<const gko::Executor> exec,
             const gko::dim<2> &size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        gpu_exec_ = std::dynamic_pointer_cast<const gko::HipExecutor>(exec);
        if (gpu_exec_ == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_descr();
    }

    HipspBase &operator=(const HipspBase &other)
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
        this->descr_ = handle_manager<hipsparseMatDescr>(
            gko::kernels::hip::hipsparse::create_mat_descr(),
            [id](hipsparseMatDescr *descr) {
                gko::device_guard g{id};
                gko::kernels::hip::hipsparse::destroy(descr);
            });
    }

private:
    std::shared_ptr<const gko::HipExecutor> gpu_exec_;
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<hipsparseMatDescr> descr_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class HipspCsr
    : public gko::EnableLinOp<HipspCsr<ValueType, IndexType>, HipspBase>,
      public gko::EnableCreateMethod<HipspCsr<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipspCsr>;
    friend class gko::EnablePolymorphicObject<HipspCsr, HipspBase>;

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
        gko::kernels::hip::hipsparse::spmv(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    HipspCsr(std::shared_ptr<const gko::Executor> exec,
            const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<HipspCsr, HipspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    hipsparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class HipspCsrmm
    : public gko::EnableLinOp<HipspCsrmm<ValueType, IndexType>, HipspBase>,
      public gko::EnableCreateMethod<HipspCsrmm<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipspCsrmm>;
    friend class gko::EnablePolymorphicObject<HipspCsrmm, HipspBase>;

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
        gko::kernels::hip::hipsparse::spmm(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
            this->get_size()[0], dense_b->get_size()[1], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            dense_b->get_size()[0], &scalars.get_const_data()[1], dx,
            dense_x->get_size()[0]);
    }

    HipspCsrmm(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<HipspCsrmm, HipspBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    hipsparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          hipsparseHybPartition_t Partition = HIPSPARSE_HYB_PARTITION_AUTO,
          int Threshold = 0>
class HipspHybrid
    : public gko::EnableLinOp<
          HipspHybrid<ValueType, IndexType, Partition, Threshold>, HipspBase>,
      public gko::EnableCreateMethod<
          HipspHybrid<ValueType, IndexType, Partition, Threshold>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipspHybrid>;
    friend class gko::EnablePolymorphicObject<HipspHybrid, HipspBase>;

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
        gko::kernels::hip::hipsparse::csr2hyb(
            this->get_gpu_exec()->get_hipsparse_handle(), this->get_size()[0],
            this->get_size()[1], this->get_descr(), t_csr->get_const_values(),
            t_csr->get_const_row_ptrs(), t_csr->get_const_col_idxs(), hyb_,
            Threshold, Partition);
    }

    ~HipspHybrid() override
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        try {
            GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseDestroyHybMat(hyb_));
        } catch (const std::exception &e) {
            std::cerr << "Error when unallocating HipspHybrid hyb_ matrix: "
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
        gko::kernels::hip::hipsparse::spmv(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
            &scalars.get_const_data()[0], this->get_descr(), hyb_, db,
            &scalars.get_const_data()[1], dx);
    }

    HipspHybrid(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<HipspHybrid, HipspBase>(exec, size),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::device_guard g{id};
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateHybMat(&hyb_));
    }

private:
    // Contains {alpha, beta}
    gko::Array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    hipsparseOperation_t trans_;
    hipsparseHybMat_t hyb_;
};


}  // namespace detail


// Some shortcuts
using hipsp_csr = detail::HipspCsr<>;
using hipsp_csrmm = detail::HipspCsrmm<>;



using hipsp_coo =
    detail::HipspHybrid<double, gko::int32, HIPSPARSE_HYB_PARTITION_USER, 0>;
using hipsp_ell =
    detail::HipspHybrid<double, gko::int32, HIPSPARSE_HYB_PARTITION_MAX, 0>;
using hipsp_hybrid = detail::HipspHybrid<>;

#endif  // GKO_BENCHMARK_SPMV_HIP_LINOPS_HPP_
