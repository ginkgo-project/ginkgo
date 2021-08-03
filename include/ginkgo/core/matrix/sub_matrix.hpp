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

#ifndef GKO_PUBLIC_CORE_MATRIX_SUBMATRIX_HPP_
#define GKO_PUBLIC_CORE_MATRIX_SUBMATRIX_HPP_


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <class MatrixType>
class SubMatrix : public EnableLinOp<SubMatrix<MatrixType>>,
                  public EnableCreateMethod<SubMatrix<MatrixType>> {
    friend class EnableCreateMethod<SubMatrix>;
    friend class EnablePolymorphicObject<SubMatrix, LinOp>;

public:
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    std::shared_ptr<MatrixType> get_sub_matrix() const { return sub_mtx_; }

    size_type get_left_overlap_bound() const { return left_overlap_bound_; }

    size_type get_left_overlap_size() const { return left_overlap_size_; }

    Array<size_type> get_overlap_sizes() const { return overlap_sizes_; }

    void convert_to(SubMatrix<MatrixType> *result) const override
    {
        this->sub_mtx_->convert_to(result->sub_mtx_.get());
        result->set_size(this->get_size());
    }

    void move_to(SubMatrix<MatrixType> *result) override
    {
        this->sub_mtx_->move_to(result->sub_mtx_.get());
        result->set_size(this->get_size());
    }

private:
    inline dim<2> compute_size(const MatrixType *matrix,
                               const gko::span &row_span,
                               const gko::span &col_span,
                               const std::vector<gko::span> &left_overlaps,
                               const std::vector<gko::span> &right_overlaps)
    {
        auto mat_size = matrix->get_size();
        size_type num_ov = 0;
        if (left_overlaps.size() > 0) {
            for (const auto &i : left_overlaps) {
                num_ov += i.length();
            }
        }
        if (right_overlaps.size() > 0) {
            for (const auto &i : right_overlaps) {
                num_ov += i.length();
            }
        }
        auto upd_size =
            gko::dim<2>(row_span.length() + num_ov, col_span.length() + num_ov);
        return upd_size;
    }

protected:
    SubMatrix(std::shared_ptr<const Executor> exec)
        : EnableLinOp<SubMatrix<MatrixType>>{exec, dim<2>{}},
          sub_mtx_{MatrixType::create(exec)},
          left_overlap_size_{0},
          left_overlaps_{},
          right_overlaps_{}
    {}

    SubMatrix(std::shared_ptr<const Executor> exec, const MatrixType *matrix,
              const gko::span &row_span, const gko::span &col_span,
              const std::vector<gko::span> &left_overlaps = {},
              const std::vector<gko::span> &right_overlaps = {})
        : EnableLinOp<SubMatrix<MatrixType>>{exec, compute_size(
                                                       matrix, row_span,
                                                       col_span, left_overlaps,
                                                       right_overlaps)},
          sub_mtx_{MatrixType::create(exec)},
          left_overlap_size_{0},
          overlap_sizes_{exec, overlap_span.size()},
          left_overlaps_{left_overlaps},
          right_overlaps_{right_overlaps}
    {
        this->generate(matrix, row_span, col_span, left_overlaps_,
                       right_overlaps_);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void apply_impl(const LinOp *b, LinOp *x,
                    const OverlapMask &write_mask) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x, const OverlapMask &write_mask) const override;

    void generate(const MatrixType *matrix, const gko::span &row_span,
                  const gko::span &col_span,
                  const std::vector<gko::span> &left_overlaps,
                  const std::vector<gko::span> &right_overlaps);

private:
    std::shared_ptr<MatrixType> sub_mtx_;
    size_type left_overlap_size_;
    size_type left_overlap_bound_{0};
    Array<size_type> overlap_sizes_;
    std::vector<gko::span> left_overlaps_;
    std::vector<gko::span> right_overlaps_;
};


}  // namespace matrix
}  // namespace gko


#endif
