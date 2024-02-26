// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <map>
#include <unordered_set>


#include <gflags/gflags.h>


#include "benchmark/sparse_blas/operations.hpp"
#include "core/base/array_access.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace {


GKO_REGISTER_OPERATION(build_lookup_offsets, csr::build_lookup_offsets);
GKO_REGISTER_OPERATION(build_lookup, csr::build_lookup);
GKO_REGISTER_OPERATION(benchmark_lookup, csr::benchmark_lookup);


}  // namespace


std::default_random_engine& get_engine();


DEFINE_int32(
    spgeam_swap_distance, 100,
    "Maximum distance for row swaps to avoid rows with disjoint column ranges");

DEFINE_string(spgemm_mode, "normal",
              R"(Which matrix B should be used to compute A * B: normal,
transposed, sparse, dense
normal: B = A for A square, A^T otherwise\ntransposed: B = A^T
sparse: B is a sparse matrix with dimensions of A^T with uniformly
        random values, at most -spgemm_rowlength non-zeros per row
dense: B is a 'dense' sparse matrix with -spgemm_rowlength columns
       and non-zeros per row)");

DEFINE_int32(spgemm_rowlength, 10,
             "The length of rows in randomly generated matrices B. Only "
             "relevant for spgemm_mode = <sparse|dense>");


using Mtx = gko::matrix::Csr<etype, itype>;


std::pair<bool, double> validate_result(gko::ptr_param<const Mtx> correct_mtx,
                                        gko::ptr_param<const Mtx> test_mtx)
{
    auto ref = gko::ReferenceExecutor::create();
    auto host_correct_mtx = gko::make_temporary_clone(ref, correct_mtx);
    auto host_test_mtx = gko::make_temporary_clone(ref, test_mtx);
    if (host_correct_mtx->get_size() != host_test_mtx->get_size() ||
        host_correct_mtx->get_num_stored_elements() !=
            host_test_mtx->get_num_stored_elements()) {
        return {false, 0.0};
    }
    double err_nrm_sq{};
    const auto size = host_correct_mtx->get_size();
    for (gko::size_type row = 0; row < size[0]; row++) {
        const auto begin = host_test_mtx->get_const_row_ptrs()[row];
        const auto end = host_test_mtx->get_const_row_ptrs()[row + 1];
        if (begin != host_correct_mtx->get_const_row_ptrs()[row] ||
            end != host_correct_mtx->get_const_row_ptrs()[row + 1] ||
            !std::equal(host_correct_mtx->get_const_col_idxs() + begin,
                        host_correct_mtx->get_const_col_idxs() + end,
                        host_test_mtx->get_const_col_idxs() + begin)) {
            return {false, 0.0};
        }
        for (auto nz = begin; nz < end; nz++) {
            const auto diff = host_test_mtx->get_const_values()[nz] -
                              host_correct_mtx->get_const_values()[nz];
            err_nrm_sq += gko::squared_norm(diff);
        }
    }
    return {true, sqrt(err_nrm_sq)};
}


class SpgemmOperation : public BenchmarkOperation {
public:
    explicit SpgemmOperation(const Mtx* mtx) : mtx_{mtx}
    {
        auto exec = mtx_->get_executor();
        const auto size = mtx_->get_size();
        std::string mode_str{FLAGS_spgemm_mode};
        if (mode_str == "normal") {
            // normal for square matrix, transposed for rectangular
            if (size[0] == size[1]) {
                mtx2_ = mtx_->clone();
            } else {
                mtx2_ = gko::as<Mtx>(mtx_->transpose());
            }
        } else if (mode_str == "transposed") {
            // always transpose
            mtx2_ = gko::as<Mtx>(mtx_->transpose());
        } else if (mode_str == "sparse") {
            // create sparse matrix of transposed size
            const auto size2 = gko::transpose(size);
            std::uniform_real_distribution<gko::remove_complex<etype>> val_dist(
                -1.0, 1.0);
            gko::matrix_data<etype, itype> data{size, {}};
            const auto local_rowlength =
                std::min<int>(FLAGS_spgemm_rowlength, size2[1]);
            data.nonzeros.reserve(size2[0] * local_rowlength);
            // randomly permute column indices
            std::vector<itype> cols(size2[1]);
            std::iota(cols.begin(), cols.end(), 0);
            for (gko::size_type row = 0; row < size2[0]; ++row) {
                std::shuffle(cols.begin(), cols.end(), get_engine());
                for (int i = 0; i < local_rowlength; ++i) {
                    data.nonzeros.emplace_back(
                        row, cols[i],
                        gko::detail::get_rand_value<etype>(val_dist,
                                                           get_engine()));
                }
            }
            data.sort_row_major();
            mtx2_ = Mtx::create(exec, size2);
            mtx2_->read(data);
        } else if (mode_str == "dense") {
            const auto size2 = gko::dim<2>(size[1], FLAGS_spgemm_rowlength);
            std::uniform_real_distribution<gko::remove_complex<etype>> dist(
                -1.0, 1.0);
            gko::matrix_data<etype, itype> data{size2, dist, get_engine()};
            data.sort_row_major();
            mtx2_ = Mtx::create(exec, size2);
            mtx2_->read(data);
        } else {
            throw gko::Error{__FILE__, __LINE__,
                             "Unsupported SpGEMM mode " + mode_str};
        }
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto correct = Mtx::create(ref, mtx_out_->get_size());
        gko::make_temporary_clone(ref, mtx_)->apply(mtx2_, correct);
        return validate_result(correct, mtx_out_);
    }

    gko::size_type get_flops() const override
    {
        auto host_exec = mtx_->get_executor()->get_master();
        auto host_mtx = gko::make_temporary_clone(host_exec, mtx_);
        auto host_mtx2 = gko::make_temporary_clone(host_exec, mtx2_);
        // count the individual products a_ik * b_kj
        gko::size_type work{};
        for (gko::size_type row = 0; row < host_mtx->get_size()[0]; row++) {
            auto begin = host_mtx->get_const_row_ptrs()[row];
            auto end = host_mtx->get_const_row_ptrs()[row + 1];
            for (auto nz = begin; nz < end; nz++) {
                auto col = host_mtx->get_const_col_idxs()[nz];
                auto local_work = host_mtx2->get_const_row_ptrs()[col + 1] -
                                  host_mtx2->get_const_row_ptrs()[col];
                work += local_work;
            }
        }
        return 2 * work;
    }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return (mtx_->get_num_stored_elements() +
                mtx2_->get_num_stored_elements() +
                mtx_out_->get_num_stored_elements()) *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override
    {
        mtx_out_ =
            Mtx::create(mtx_->get_executor(),
                        gko::dim<2>{mtx_->get_size()[0], mtx2_->get_size()[1]});
    }

    void run() override { mtx_->apply(mtx2_, mtx_out_); }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx2_;
    std::unique_ptr<Mtx> mtx_out_;
};


class SpgeamOperation : public BenchmarkOperation {
public:
    explicit SpgeamOperation(const Mtx* mtx) : mtx_{mtx}
    {
        auto exec = mtx_->get_executor();
        const auto size = mtx_->get_size();
        // randomly permute n/2 rows with limited distances
        gko::array<itype> permutation_array(exec->get_master(), size[0]);
        auto permutation = permutation_array.get_data();
        std::iota(permutation, permutation + size[0], 0);
        std::uniform_int_distribution<itype> start_dist(0, size[0] - 1);
        std::uniform_int_distribution<itype> delta_dist(
            -FLAGS_spgeam_swap_distance, FLAGS_spgeam_swap_distance);
        for (itype i = 0; i < size[0] / 2; ++i) {
            auto a = start_dist(get_engine());
            auto b = a + delta_dist(get_engine());
            if (b >= 0 && b < size[0]) {
                std::swap(permutation[a], permutation[b]);
            }
        }
        mtx2_ = gko::as<Mtx>(mtx_->row_permute(&permutation_array));
        id_ = gko::matrix::Identity<etype>::create(exec, size[1]);
        scalar_ = gko::initialize<gko::matrix::Dense<etype>>({1.0}, exec);
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto correct = gko::make_temporary_clone(ref, mtx2_);
        gko::make_temporary_clone(ref, mtx_)->apply(scalar_, id_, scalar_,
                                                    correct.get());
        return validate_result(correct.get(), mtx_out_);
    }

    gko::size_type get_flops() const override
    {
        return mtx_->get_num_stored_elements() +
               mtx2_->get_num_stored_elements();
    }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return (mtx_->get_num_stored_elements() +
                mtx2_->get_num_stored_elements() +
                mtx_out_->get_num_stored_elements()) *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override { mtx_out_ = mtx2_->clone(); }

    void run() override { mtx_->apply(scalar_, id_, scalar_, mtx_out_); }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx2_;
    std::unique_ptr<gko::matrix::Dense<etype>> scalar_;
    std::unique_ptr<gko::matrix::Identity<etype>> id_;
    std::unique_ptr<Mtx> mtx_out_;
};


class TransposeOperation : public BenchmarkOperation {
public:
    explicit TransposeOperation(const Mtx* mtx) : mtx_{mtx} {}

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        return validate_result(
            gko::as<Mtx>(gko::make_temporary_clone(ref, mtx_)->transpose()),
            mtx_out_);
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return 2 * mtx_->get_num_stored_elements() *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override { mtx_out_ = nullptr; }

    void run() override { mtx_out_ = gko::as<Mtx>(mtx_->transpose()); }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx_out_;
};


class SortOperation : public BenchmarkOperation {
public:
    explicit SortOperation(const Mtx* mtx)
    {
        mtx_shuffled_ = mtx->clone();
        gko::test::unsort_matrix(mtx_shuffled_, get_engine());
        mtx_out_ = mtx_shuffled_->clone();
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto mtx_sorted = gko::clone(ref, mtx_shuffled_);
        mtx_sorted->sort_by_column_index();
        return validate_result(mtx_sorted, mtx_out_);
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, read row pointers once
        return 2 * mtx_shuffled_->get_num_stored_elements() *
                   (sizeof(etype) + sizeof(itype)) +
               mtx_shuffled_->get_size()[0] * sizeof(itype);
    }

    void prepare() override { mtx_out_->copy_from(mtx_shuffled_); }

    void run() override
    {
        // FIXME: here, we are measuring sorted input except for the first run
        mtx_out_->sort_by_column_index();
    }

private:
    std::unique_ptr<Mtx> mtx_shuffled_;
    std::unique_ptr<Mtx> mtx_out_;
};


class IsSortedOperation : public BenchmarkOperation {
public:
    explicit IsSortedOperation(const Mtx* mtx) : mtx_{mtx}, result_{} {}

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto mtx_sorted = gko::make_temporary_clone(ref, mtx_);
        return {mtx_sorted->is_sorted_by_column_index() == result_, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read sparsity pattern and row pointers once
        return (mtx_->get_num_stored_elements() + mtx_->get_size()[0]) *
               sizeof(itype);
    }

    void run() override { result_ = mtx_->is_sorted_by_column_index(); }

private:
    const Mtx* mtx_;
    bool result_;
};


DEFINE_bool(lookup_no_full, false,
            "Disable lookup specialization for rows with contiguous non-zero "
            "range in lookup and generate_lookup benchmark");

DEFINE_bool(lookup_no_bitmap, false,
            "Disable lookup specialization using a bitmap in lookup and "
            "generate_lookup benchmark");


gko::matrix::csr::sparsity_type get_allowed_sparsity()
{
    auto allowed_sparsity = gko::matrix::csr::sparsity_type::hash;
    if (!FLAGS_lookup_no_full) {
        allowed_sparsity =
            allowed_sparsity | gko::matrix::csr::sparsity_type::full;
    }
    if (!FLAGS_lookup_no_bitmap) {
        allowed_sparsity =
            allowed_sparsity | gko::matrix::csr::sparsity_type::bitmap;
    }
    return allowed_sparsity;
}


class GenerateLookupOperation : public BenchmarkOperation {
public:
    explicit GenerateLookupOperation(const Mtx* mtx)
        : mtx_{mtx},
          allowed_sparsity_{get_allowed_sparsity()},
          storage_offsets_{mtx->get_executor(), mtx->get_size()[0] + 1},
          row_descs_{mtx->get_executor(), mtx->get_size()[0]},
          storage_{mtx->get_executor()}
    {
        const auto exec = mtx_->get_executor();
        const auto num_rows = mtx_->get_size()[0];
        exec->run(make_build_lookup_offsets(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(), num_rows,
            allowed_sparsity_, storage_offsets_.get_data()));
        storage_.resize_and_reset(get_element(storage_offsets_, num_rows));
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto host_mtx = gko::make_temporary_clone(ref, mtx_);
        gko::array<itype> host_storage_offsets{ref, storage_offsets_};
        gko::array<gko::int64> host_row_descs{ref, row_descs_};
        gko::array<gko::int32> host_storage{ref, storage_};
        const auto row_ptrs = host_mtx->get_const_row_ptrs();
        const auto col_idxs = host_mtx->get_const_col_idxs();
        for (gko::size_type row = 0; row < mtx_->get_size()[0]; row++) {
            gko::matrix::csr::device_sparsity_lookup<itype> lookup{
                row_ptrs,
                col_idxs,
                host_storage_offsets.get_const_data(),
                host_storage.get_const_data(),
                host_row_descs.get_const_data(),
                row};
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            for (auto nz = begin; nz < end; nz++) {
                const auto col = col_idxs[nz];
                if (lookup.lookup_unsafe(col) + begin != nz) {
                    return {false, 0.0};
                }
            }
        }
        return {true, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read sparsity pattern and row pointers once, write lookup structures
        return mtx_->get_num_stored_elements() * sizeof(itype) +
               mtx_->get_size()[0] * (2 * sizeof(itype) + sizeof(gko::int64)) +
               storage_.get_size() * sizeof(gko::int32);
    }

    void run() override
    {
        const auto exec = mtx_->get_executor();
        const auto num_rows = mtx_->get_size()[0];
        exec->run(make_build_lookup_offsets(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(), num_rows,
            allowed_sparsity_, storage_offsets_.get_data()));
        exec->run(make_build_lookup(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(), num_rows,
            allowed_sparsity_, storage_offsets_.get_const_data(),
            row_descs_.get_data(), storage_.get_data()));
    }

private:
    const Mtx* mtx_;
    gko::matrix::csr::sparsity_type allowed_sparsity_;
    gko::array<itype> storage_offsets_;
    gko::array<gko::int64> row_descs_;
    gko::array<gko::int32> storage_;
};


DEFINE_int32(lookup_sample_size, 10,
             "Number of elements to look up from each row in lookup benchmark");


class LookupOperation : public BenchmarkOperation {
public:
    explicit LookupOperation(const Mtx* mtx)
        : mtx_{mtx},
          allowed_sparsity_{get_allowed_sparsity()},
          storage_offsets_{mtx->get_executor(), mtx->get_size()[0] + 1},
          row_descs_{mtx->get_executor(), mtx->get_size()[0]},
          storage_{mtx->get_executor()},
          results_{mtx->get_executor()}
    {
        sample_size_ = 10;
        const auto exec = mtx_->get_executor();
        const auto num_rows = mtx_->get_size()[0];
        results_.resize_and_reset(num_rows * sample_size_);
        exec->run(make_build_lookup_offsets(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(), num_rows,
            allowed_sparsity_, storage_offsets_.get_data()));
        storage_.resize_and_reset(get_element(storage_offsets_, num_rows));
        exec->run(make_build_lookup(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(), num_rows,
            allowed_sparsity_, storage_offsets_.get_const_data(),
            row_descs_.get_data(), storage_.get_data()));
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto host_mtx = gko::make_temporary_clone(ref, mtx_);
        gko::array<itype> host_results{ref, results_};
        const auto row_ptrs = host_mtx->get_const_row_ptrs();
        for (gko::size_type row = 0; row < mtx_->get_size()[0]; row++) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto row_len = row_end - row_begin;
            for (itype sample = 0; sample < sample_size_; sample++) {
                const auto expected =
                    row_len > 0 ? row_len * sample / sample_size_ + row_begin
                                : -1;
                if (host_results.get_const_data()[row * sample_size_ +
                                                  sample] != expected) {
                    return {false, 0.0};
                }
            }
        }
        return {true, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read row pointers and lookup structures, for each sample read a
        // column index and write a result
        return mtx_->get_size()[0] * (2 * sizeof(itype) + sizeof(gko::int64) +
                                      sample_size_ * 2 * sizeof(itype)) +
               storage_.get_size() * sizeof(gko::int32);
    }

    void run() override
    {
        mtx_->get_executor()->run(make_benchmark_lookup(
            mtx_->get_const_row_ptrs(), mtx_->get_const_col_idxs(),
            mtx_->get_size()[0], storage_offsets_.get_const_data(),
            row_descs_.get_data(), storage_.get_data(), sample_size_,
            results_.get_data()));
    }

private:
    const Mtx* mtx_;
    gko::matrix::csr::sparsity_type allowed_sparsity_;
    itype sample_size_;
    gko::array<itype> storage_offsets_;
    gko::array<gko::int64> row_descs_;
    gko::array<gko::int32> storage_;
    gko::array<itype> results_;
};


bool validate_symbolic_factorization(const Mtx* input, const Mtx* factors)
{
    const auto host_exec = input->get_executor()->get_master();
    const auto host_input = gko::make_temporary_clone(host_exec, input);
    const auto host_factors = gko::make_temporary_clone(host_exec, factors);
    const auto num_rows = input->get_size()[0];
    const auto in_row_ptrs = host_input->get_const_row_ptrs();
    const auto in_cols = host_input->get_const_col_idxs();
    const auto factor_row_ptrs = host_factors->get_const_row_ptrs();
    const auto factor_cols = host_factors->get_const_col_idxs();
    std::unordered_set<itype> columns;
    for (itype row = 0; row < num_rows; row++) {
        const auto in_begin = in_cols + in_row_ptrs[row];
        const auto in_end = in_cols + in_row_ptrs[row + 1];
        const auto factor_begin = factor_cols + factor_row_ptrs[row];
        const auto factor_end = factor_cols + factor_row_ptrs[row + 1];
        columns.clear();
        // the factor needs to contain the original matrix
        // plus the diagonal if that was missing
        columns.insert(in_begin, in_end);
        columns.insert(row);
        for (auto col_it = factor_begin; col_it < factor_end; ++col_it) {
            const auto col = *col_it;
            if (col >= row) {
                break;
            }
            const auto dep_begin = factor_cols + factor_row_ptrs[col];
            const auto dep_end = factor_cols + factor_row_ptrs[col + 1];
            // insert the upper triangular part of the row
            const auto dep_diag = std::find(dep_begin, dep_end, col);
            columns.insert(dep_diag, dep_end);
        }
        // the factor should contain exactly these columns, no more
        if (factor_end - factor_begin != columns.size()) {
            return false;
        }
        for (auto col_it = factor_begin; col_it < factor_end; ++col_it) {
            if (columns.find(*col_it) == columns.end()) {
                return false;
            }
        }
    }
    return true;
}


class SymbolicLuOperation : public BenchmarkOperation {
public:
    explicit SymbolicLuOperation(const Mtx* mtx) : mtx_{mtx}, result_{} {}

    std::pair<bool, double> validate() const override
    {
        return std::make_pair(
            validate_symbolic_factorization(mtx_, result_.get()), 0.0);
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void run() override { gko::factorization::symbolic_lu(mtx_, result_); }

    void write_stats(json& object) override
    {
        object["factor_nonzeros"] = result_->get_num_stored_elements();
    }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> result_;
};


class SymbolicLuNearSymmOperation : public BenchmarkOperation {
public:
    explicit SymbolicLuNearSymmOperation(const Mtx* mtx) : mtx_{mtx}, result_{}
    {}

    std::pair<bool, double> validate() const override
    {
        return std::make_pair(
            validate_symbolic_factorization(mtx_, result_.get()), 0.0);
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void run() override
    {
        gko::factorization::symbolic_lu_near_symm(mtx_, result_);
    }

    void write_stats(json& object) override
    {
        object["factor_nonzeros"] = result_->get_num_stored_elements();
    }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> result_;
};


class SymbolicCholeskyOperation : public BenchmarkOperation {
public:
    explicit SymbolicCholeskyOperation(const Mtx* mtx, bool symmetric)
        : mtx_{mtx}, symmetric_{symmetric}, result_{}
    {}

    std::pair<bool, double> validate() const override
    {
        if (symmetric_) {
            return std::make_pair(
                validate_symbolic_factorization(mtx_, result_.get()), 0.0);
        } else {
            const auto exec = mtx_->get_executor();
            const auto symm_result = result_->clone();
            const auto lt_factor = gko::as<Mtx>(symm_result->transpose());
            const auto scalar = gko::initialize<gko::matrix::Dense<etype>>(
                {gko::one<etype>()}, exec);
            const auto id =
                gko::matrix::Identity<etype>::create(exec, mtx_->get_size()[0]);
            lt_factor->apply(scalar, id, scalar, symm_result);
            return std::make_pair(
                validate_symbolic_factorization(mtx_, symm_result.get()), 0.0);
        }
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void run() override
    {
        gko::factorization::symbolic_cholesky(mtx_, symmetric_, result_,
                                              forest_);
    }

    void write_stats(json& object) override
    {
        object["factor_nonzeros"] = result_->get_num_stored_elements();
    }

private:
    const Mtx* mtx_;
    bool symmetric_;
    std::unique_ptr<Mtx> result_;
    std::unique_ptr<gko::factorization::elimination_forest<itype>> forest_;
};


class ReorderRcmOperation : public BenchmarkOperation {
    using reorder_type = gko::experimental::reorder::Rcm<itype>;
    using permute_type = gko::matrix::Permutation<itype>;

public:
    explicit ReorderRcmOperation(const Mtx* mtx)
        : mtx_{mtx->clone()},
          factory_{reorder_type::build().on(mtx->get_executor())}
    {}

    std::pair<bool, double> validate() const override
    {
        // validating RCM correctness is hard, let's leave it out for now
        return {true, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void prepare() override {}

    void run() override { reorder_ = factory_->generate(mtx_); }

private:
    std::shared_ptr<Mtx> mtx_;
    std::unique_ptr<reorder_type> factory_;
    std::unique_ptr<permute_type> reorder_;
};


#if GKO_HAVE_METIS


class ReorderNestedDissectionOperation : public BenchmarkOperation {
    using factory_type =
        gko::experimental::reorder::NestedDissection<etype, itype>;
    using reorder_type = gko::matrix::Permutation<itype>;

public:
    explicit ReorderNestedDissectionOperation(const Mtx* mtx)
        : mtx_{mtx->clone()},
          factory_{factory_type::build().on(mtx->get_executor())}
    {}

    std::pair<bool, double> validate() const override
    {
        // validating ND correctness is hard, let's leave it out for now
        return {true, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void prepare() override {}

    void run() override { reorder_ = factory_->generate(mtx_); }

private:
    std::shared_ptr<Mtx> mtx_;
    std::unique_ptr<factory_type> factory_;
    std::unique_ptr<reorder_type> reorder_;
};


#endif


class ReorderApproxMinDegOperation : public BenchmarkOperation {
    using factory_type = gko::experimental::reorder::Amd<itype>;
    using reorder_type = gko::matrix::Permutation<itype>;

public:
    explicit ReorderApproxMinDegOperation(const Mtx* mtx)
        : mtx_{mtx->clone()},
          factory_{factory_type::build().on(mtx->get_executor())}
    {}

    std::pair<bool, double> validate() const override
    {
        // validating AMD correctness is hard, let's leave it out for now
        return {true, 0.0};
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override { return 0; }

    void prepare() override {}

    void run() override { reorder_ = factory_->generate(mtx_); }

private:
    std::shared_ptr<Mtx> mtx_;
    std::unique_ptr<factory_type> factory_;
    std::unique_ptr<reorder_type> reorder_;
};


const std::map<std::string,
               std::function<std::unique_ptr<BenchmarkOperation>(const Mtx*)>>
    operation_map{
        {"spgemm",
         [](const Mtx* mtx) { return std::make_unique<SpgemmOperation>(mtx); }},
        {"spgeam",
         [](const Mtx* mtx) { return std::make_unique<SpgeamOperation>(mtx); }},
        {"transpose",
         [](const Mtx* mtx) {
             return std::make_unique<TransposeOperation>(mtx);
         }},
        {"sort",
         [](const Mtx* mtx) { return std::make_unique<SortOperation>(mtx); }},
        {"is_sorted",
         [](const Mtx* mtx) {
             return std::make_unique<IsSortedOperation>(mtx);
         }},
        {"generate_lookup",
         [](const Mtx* mtx) {
             return std::make_unique<GenerateLookupOperation>(mtx);
         }},
        {"lookup",
         [](const Mtx* mtx) { return std::make_unique<LookupOperation>(mtx); }},
        {"symbolic_lu",
         [](const Mtx* mtx) {
             return std::make_unique<SymbolicLuOperation>(mtx);
         }},
        {"symbolic_lu_near_symm",
         [](const Mtx* mtx) {
             return std::make_unique<SymbolicLuNearSymmOperation>(mtx);
         }},
        {"symbolic_cholesky",
         [](const Mtx* mtx) {
             return std::make_unique<SymbolicCholeskyOperation>(mtx, false);
         }},
        {"symbolic_cholesky_symmetric",
         [](const Mtx* mtx) {
             return std::make_unique<SymbolicCholeskyOperation>(mtx, true);
         }},
        {"reorder_rcm",
         [](const Mtx* mtx) {
             return std::make_unique<ReorderRcmOperation>(mtx);
         }},
        {"reorder_amd",
         [](const Mtx* mtx) {
             return std::make_unique<ReorderApproxMinDegOperation>(mtx);
         }},
        {"reorder_nd",
         [](const Mtx* mtx) -> std::unique_ptr<BenchmarkOperation> {
#if GKO_HAVE_METIS
             return std::make_unique<ReorderNestedDissectionOperation>(mtx);
#else
             GKO_NOT_COMPILED(METIS);
#endif
         }}};


std::unique_ptr<BenchmarkOperation> get_operation(std::string name,
                                                  const Mtx* matrix)
{
    return operation_map.at(name)(matrix);
}
