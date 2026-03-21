// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/mumps.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <cmumps_c.h>
#include <dmumps_c.h>
#include <smumps_c.h>
#include <zmumps_c.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace solver {
namespace {


// Job codes for MUMPS
constexpr int mumps_job_init = -1;
constexpr int mumps_job_end = -2;
constexpr int mumps_job_analyze = 1;
constexpr int mumps_job_factorize = 2;
constexpr int mumps_job_solve = 3;


// Traits to map Ginkgo ValueType to the corresponding MUMPS struct
template <typename ValueType>
struct mumps_traits;

template <>
struct mumps_traits<float> {
    using mumps_struct = SMUMPS_STRUC_C;
    static void mumps_c(mumps_struct* data) { smumps_c(data); }
};

template <>
struct mumps_traits<double> {
    using mumps_struct = DMUMPS_STRUC_C;
    static void mumps_c(mumps_struct* data) { dmumps_c(data); }
};

template <>
struct mumps_traits<std::complex<float>> {
    using mumps_struct = CMUMPS_STRUC_C;
    static void mumps_c(mumps_struct* data) { cmumps_c(data); }
};

template <>
struct mumps_traits<std::complex<double>> {
    using mumps_struct = ZMUMPS_STRUC_C;
    static void mumps_c(mumps_struct* data) { zmumps_c(data); }
};


// Helper to get the element pointer for MUMPS (real types return pointer
// directly, complex types need a cast to mumps_complex/mumps_double_complex)
template <typename ValueType>
auto mumps_value_ptr(ValueType* ptr) -> std::remove_pointer_t<
    decltype(std::declval<typename mumps_traits<ValueType>::mumps_struct>().a)>*
{
    using mumps_struct = typename mumps_traits<ValueType>::mumps_struct;
    using mumps_val_type =
        std::remove_pointer_t<decltype(std::declval<mumps_struct>().a)>;
    return reinterpret_cast<mumps_val_type*>(ptr);
}


}  // namespace


// The opaque state holding the MUMPS instance
template <typename ValueType, typename IndexType>
struct Mumps<ValueType, IndexType>::mumps_state {
    using traits = mumps_traits<ValueType>;
    using mumps_struct = typename traits::mumps_struct;

    mumps_struct data;
    bool initialized = false;

    // COO format storage for MUMPS (1-indexed)
    std::vector<int> irn;
    std::vector<int> jcn;
    std::vector<ValueType> a;

    mumps_state() = default;

    ~mumps_state() { finalize(); }

    mumps_state(const mumps_state&) = delete;
    mumps_state& operator=(const mumps_state&) = delete;
    mumps_state(mumps_state&&) = delete;
    mumps_state& operator=(mumps_state&&) = delete;

    void initialize(bool symmetric)
    {
        if (initialized) {
            finalize();
        }
        // Use MPI_COMM_SELF so each rank solves independently.
        // NOTE: MUMPS is Fortran-based and requires the Fortran MPI
        // runtime (libmpifort) and ScaLAPACK to be linked. Without
        // them, MPI_Comm_c2f handles won't resolve correctly in the
        // Fortran layer, causing "Instance Error 1".
        std::memset(&data, 0, sizeof(data));
        data.par = 1;                  // host participates in factorization
        data.sym = symmetric ? 2 : 0;  // 0 = unsymmetric, 2 = symmetric
        data.comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
        data.job = mumps_job_init;
        traits::mumps_c(&data);
        initialized = true;

        // Suppress all MUMPS output
        data.icntl[0] = -1;   // no error messages
        data.icntl[1] = -1;   // no diagnostic output
        data.icntl[2] = -1;   // no global info
        data.icntl[3] = 0;    // verbosity level off
        data.icntl[13] = 50;  // ICNTL(14): % memory relaxation (default 20)
    }

    void finalize()
    {
        if (initialized) {
            data.job = mumps_job_end;
            traits::mumps_c(&data);
            initialized = false;
        }
    }

    void set_ordering(int ordering) { data.icntl[6] = ordering; }

    void set_matrix(int n, int nnz, int* irn_ptr, int* jcn_ptr,
                    ValueType* a_ptr)
    {
        data.n = n;
        data.nnz = nnz;
        data.irn = irn_ptr;
        data.jcn = jcn_ptr;
        data.a = mumps_value_ptr(a_ptr);
    }

    void analyze()
    {
        data.job = mumps_job_analyze;
        traits::mumps_c(&data);
        GKO_THROW_IF_INVALID(data.infog[0] >= 0,
                             "MUMPS analysis failed with INFOG(1) = " +
                                 std::to_string(data.infog[0]));
    }

    void factorize()
    {
        data.job = mumps_job_factorize;
        traits::mumps_c(&data);
        GKO_THROW_IF_INVALID(data.infog[0] >= 0,
                             "MUMPS factorization failed with INFOG(1) = " +
                                 std::to_string(data.infog[0]));
    }

    /**
     * Solves the system. rhs_ptr must point to a column-major array of
     * size (ldrhs, nrhs) where ldrhs >= n.
     */
    void solve(int nrhs, ValueType* rhs_ptr, int ldrhs)
    {
        data.nrhs = nrhs;
        data.lrhs = ldrhs;
        data.rhs = mumps_value_ptr(rhs_ptr);
        data.job = mumps_job_solve;
        traits::mumps_c(&data);
        GKO_THROW_IF_INVALID(data.infog[0] >= 0,
                             "MUMPS solve failed with INFOG(1) = " +
                                 std::to_string(data.infog[0]));
    }
};


// Helper to convert CSR to COO with 1-based indexing.
// If lower_only is true, only entries where row >= col are kept
// (lower triangular including diagonal). This is required for
// MUMPS symmetric mode (sym=1 or sym=2).
template <typename IndexType, typename ValueType>
void csr_to_coo_1based(int num_rows, const IndexType* row_ptrs,
                       const IndexType* col_idxs, const ValueType* values,
                       std::vector<int>& irn, std::vector<int>& jcn,
                       std::vector<ValueType>& a, bool lower_only)
{
    irn.clear();
    jcn.clear();
    a.clear();
    for (int row = 0; row < num_rows; ++row) {
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            const auto col = col_idxs[idx];
            if (lower_only && col > row) {
                continue;
            }
            irn.push_back(row + 1);  // 1-based
            jcn.push_back(col + 1);  // 1-based
            a.push_back(values[idx]);
        }
    }
}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>::Mumps(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Mumps>(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>::Mumps(const Factory* factory,
                                   std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<Mumps>(factory->get_executor(), system_matrix->get_size()),
      parameters_{factory->get_parameters()},
      state_(std::make_unique<mumps_state>())
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // Convert to CSR on host
    auto host_csr = copy_and_convert_to<matrix_type>(host_exec, system_matrix);
    system_matrix_ = std::move(host_csr);

    const auto num_rows = static_cast<int>(system_matrix_->get_size()[0]);

    csr_to_coo_1based(num_rows, system_matrix_->get_const_row_ptrs(),
                      system_matrix_->get_const_col_idxs(),
                      system_matrix_->get_const_values(), state_->irn,
                      state_->jcn, state_->a, parameters_.symmetric);

    const auto nnz = static_cast<int>(state_->irn.size());

    // Initialize MUMPS and run analysis + factorization
    state_->initialize(parameters_.symmetric);
    state_->set_ordering(parameters_.ordering);
    state_->set_matrix(num_rows, nnz, state_->irn.data(), state_->jcn.data(),
                       state_->a.data());
    state_->analyze();
    state_->factorize();
}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>::~Mumps() = default;


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>::Mumps(const Mumps& other)
    : EnableLinOp<Mumps>(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>::Mumps(Mumps&& other) noexcept
    : EnableLinOp<Mumps>(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>& Mumps<ValueType, IndexType>::operator=(
    const Mumps& other)
{
    if (this != &other) {
        EnableLinOp<Mumps>::operator=(other);
        parameters_ = other.parameters_;
        system_matrix_ = other.system_matrix_;
        // Re-factorize from stored matrix
        if (other.state_ && other.state_->initialized) {
            state_ = std::make_unique<mumps_state>();
            const auto num_rows =
                static_cast<int>(system_matrix_->get_size()[0]);

            csr_to_coo_1based(num_rows, system_matrix_->get_const_row_ptrs(),
                              system_matrix_->get_const_col_idxs(),
                              system_matrix_->get_const_values(), state_->irn,
                              state_->jcn, state_->a, parameters_.symmetric);

            const auto nnz = static_cast<int>(state_->irn.size());

            state_->initialize(parameters_.symmetric);
            state_->set_ordering(parameters_.ordering);
            state_->set_matrix(num_rows, nnz, state_->irn.data(),
                               state_->jcn.data(), state_->a.data());
            state_->analyze();
            state_->factorize();
        } else {
            state_.reset();
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Mumps<ValueType, IndexType>& Mumps<ValueType, IndexType>::operator=(
    Mumps&& other) noexcept
{
    if (this != &other) {
        EnableLinOp<Mumps>::operator=(std::move(other));
        parameters_ = std::exchange(other.parameters_, {});
        system_matrix_ = std::move(other.system_matrix_);
        state_ = std::move(other.state_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
void Mumps<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            const auto host_exec = this->get_executor()->get_master();
            const auto size = dense_b->get_size();
            const auto num_rows = static_cast<int>(size[0]);

            // Ensure persistent host buffer is allocated with
            // contiguous storage (stride = ncols). Dense::copy_from
            // adopts the source's stride, so we must re-create the
            // buffer whenever the stride would change.
            if (!host_buffer_.get() || host_buffer_->get_size() != size ||
                host_buffer_->get_stride() != size[1]) {
                host_buffer_.vec =
                    matrix::Dense<ValueType>::create(host_exec, size);
            }

            // Copy b row-by-row into the contiguous host buffer.
            // We cannot use copy_from because it adopts the source
            // stride, which may come from a Dense submatrix view.
            auto host_b = make_temporary_clone(host_exec, dense_b);
            for (int i = 0; i < num_rows; ++i) {
                for (size_type j = 0; j < size[1]; ++j) {
                    host_buffer_->at(i, j) = host_b->at(i, j);
                }
            }

            // MUMPS expects column-major RHS with lrhs >= n.
            // For a single RHS the layout is equivalent, just set
            // lrhs = n.
            state_->solve(1, host_buffer_->get_values(), num_rows);

            // Copy solution row-by-row back to x (which may be a
            // strided Dense view).
            auto host_x = make_temporary_clone(host_exec, dense_x);
            for (int i = 0; i < num_rows; ++i) {
                for (size_type j = 0; j < size[1]; ++j) {
                    host_x->at(i, j) = host_buffer_->at(i, j);
                }
            }
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Mumps<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_MUMPS(ValueType, IndexType) \
    class Mumps<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_MUMPS);


}  // namespace solver
}  // namespace experimental
}  // namespace gko
