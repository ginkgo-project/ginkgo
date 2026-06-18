// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/hypre_boomeramg.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>

#include <_hypre_parcsr_mv.h>
#include <_hypre_utilities.h>
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace solver {
namespace {


// Ensure HYPRE is initialized exactly once. If another library (e.g.
// PETSc) already initialized HYPRE, we skip the call.
// We intentionally never call HYPRE_Finalize() — a library should not
// finalize a shared resource since the destruction order of objects
// across libraries (especially in Python with GC) is unpredictable.
// The OS reclaims all memory at process exit.
std::once_flag hypre_init_flag;

void ensure_hypre_init()
{
    std::call_once(hypre_init_flag, []() {
        if (!HYPRE_Initialized()) {
            HYPRE_Init();
        }
    });
}


// HYPRE functions return a nonzero error flag (a bitmask of HYPRE_ERROR_*)
// on failure. We don't treat these as fatal — BoomerAMG is used here as a
// preconditioner and a failed/non-converged cycle should not abort the
// host application — but we must not swallow them silently either. Warn,
// then clear the flag so it doesn't leak into unrelated later HYPRE calls.
void warn_on_hypre_error(HYPRE_Int error, const char* call,
                         HYPRE_Int ignore_mask = 0)
{
    // Clear the (background) flags we deliberately tolerate before deciding
    // whether anything is left worth warning about.
    error &= ~ignore_mask;
    if (error != 0) {
        char description[256] = {};
        HYPRE_DescribeError(error, description);
        std::cerr << "Ginkgo warning: HypreBoomerAmg: HYPRE call failed ("
                  << call << "): error code " << error << " (" << description
                  << ")" << std::endl;
    }
    // Always clear so a flag does not leak into unrelated later HYPRE calls.
    HYPRE_ClearAllErrors();
}


}  // namespace


#define GKO_CHECK_HYPRE(_call) warn_on_hypre_error((_call), #_call)
// Single-cycle preconditioner solves never reach a tolerance, so HYPRE's
// convergence flag is expected and must not be reported as a failure.
#define GKO_CHECK_HYPRE_SOLVE(_call) \
    warn_on_hypre_error((_call), #_call, HYPRE_ERROR_CONV)


template <typename ValueType>
struct HypreBoomerAmg<ValueType>::hypre_state {
    HYPRE_IJMatrix ij_matrix = nullptr;
    HYPRE_ParCSRMatrix par_matrix = nullptr;
    HYPRE_IJVector ij_rhs = nullptr;
    HYPRE_ParVector par_rhs = nullptr;
    HYPRE_IJVector ij_sol = nullptr;
    HYPRE_ParVector par_sol = nullptr;
    HYPRE_Solver amg_solver = nullptr;
    HYPRE_Int num_rows = 0;

    // Owned buffers used only when we cannot view Ginkgo data directly
    // (e.g. float -> double conversion, or strided multi-column data).
    std::vector<HYPRE_Real> rhs_buffer;
    std::vector<HYPRE_Real> sol_buffer;

    hypre_state() = default;

    ~hypre_state() { destroy(); }

    hypre_state(const hypre_state&) = delete;
    hypre_state& operator=(const hypre_state&) = delete;
    hypre_state(hypre_state&&) = delete;
    hypre_state& operator=(hypre_state&&) = delete;

    /**
     * Returns a reference to the raw data pointer of the local part of a
     * HYPRE_ParVector, allowing it to be redirected to external storage.
     */
    static HYPRE_Real*& vector_data_ref(HYPRE_ParVector pv)
    {
        return hypre_VectorData(hypre_ParVectorLocalVector(pv));
    }

    void destroy()
    {
        // If HYPRE has already been finalized (e.g. by PetscFinalize()),
        // its internal memory pools are gone. Calling any HYPRE_*Destroy
        // function would access freed memory. In that case, just null out
        // our handles and let the OS reclaim the leaked memory at exit.
        if (!HYPRE_Initialized()) {
            amg_solver = nullptr;
            ij_rhs = nullptr;
            ij_sol = nullptr;
            ij_matrix = nullptr;
            par_matrix = nullptr;
            par_rhs = nullptr;
            par_sol = nullptr;
            return;
        }

        if (amg_solver) {
            GKO_CHECK_HYPRE(HYPRE_BoomerAMGDestroy(amg_solver));
            amg_solver = nullptr;
        }
        // Data owner is 0, so HYPRE won't free the data pointer.
        // Just restore our owned buffers so the std::vector destructor
        // frees the right memory.
        if (ij_rhs) {
            vector_data_ref(par_rhs) = rhs_buffer.data();
            GKO_CHECK_HYPRE(HYPRE_IJVectorDestroy(ij_rhs));
            ij_rhs = nullptr;
        }
        if (ij_sol) {
            vector_data_ref(par_sol) = sol_buffer.data();
            GKO_CHECK_HYPRE(HYPRE_IJVectorDestroy(ij_sol));
            ij_sol = nullptr;
        }
        if (ij_matrix) {
            GKO_CHECK_HYPRE(HYPRE_IJMatrixDestroy(ij_matrix));
            ij_matrix = nullptr;
        }
        par_matrix = nullptr;
        par_rhs = nullptr;
        par_sol = nullptr;
    }

    void create_matrix(
        HYPRE_Int n,
        const std::shared_ptr<const matrix::Csr<double, int32>>& csr)
    {
        num_rows = n;

        // Create IJ matrix (sequential: rows [0, n))
        GKO_CHECK_HYPRE(
            HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, n - 1, 0, n - 1, &ij_matrix));
        GKO_CHECK_HYPRE(
            HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR));
        GKO_CHECK_HYPRE(HYPRE_IJMatrixInitialize(ij_matrix));

        const auto row_ptrs = csr->get_const_row_ptrs();
        const auto col_idxs = csr->get_const_col_idxs();
        const auto values = csr->get_const_values();

        // Set values row by row
        for (HYPRE_Int row = 0; row < n; ++row) {
            HYPRE_Int row_start = row_ptrs[row];
            HYPRE_Int row_end = row_ptrs[row + 1];
            HYPRE_Int ncols = row_end - row_start;
            std::vector<HYPRE_Int> cols(ncols);
            for (HYPRE_Int k = 0; k < ncols; ++k) {
                cols[k] = static_cast<HYPRE_Int>(col_idxs[row_start + k]);
            }
            GKO_CHECK_HYPRE(HYPRE_IJMatrixSetValues(ij_matrix, 1, &ncols, &row,
                                                    cols.data(),
                                                    values + row_start));
        }

        GKO_CHECK_HYPRE(HYPRE_IJMatrixAssemble(ij_matrix));
        GKO_CHECK_HYPRE(HYPRE_IJMatrixGetObject(
            ij_matrix, reinterpret_cast<void**>(&par_matrix)));
    }

    void create_vectors(HYPRE_Int n)
    {
        // Allocate owned buffers
        rhs_buffer.resize(n, 0.0);
        sol_buffer.resize(n, 0.0);

        // RHS vector
        GKO_CHECK_HYPRE(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n - 1, &ij_rhs));
        GKO_CHECK_HYPRE(HYPRE_IJVectorSetObjectType(ij_rhs, HYPRE_PARCSR));
        GKO_CHECK_HYPRE(HYPRE_IJVectorInitialize(ij_rhs));
        GKO_CHECK_HYPRE(HYPRE_IJVectorAssemble(ij_rhs));
        GKO_CHECK_HYPRE(
            HYPRE_IJVectorGetObject(ij_rhs, reinterpret_cast<void**>(&par_rhs)));

        // Solution vector
        GKO_CHECK_HYPRE(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n - 1, &ij_sol));
        GKO_CHECK_HYPRE(HYPRE_IJVectorSetObjectType(ij_sol, HYPRE_PARCSR));
        GKO_CHECK_HYPRE(HYPRE_IJVectorInitialize(ij_sol));
        GKO_CHECK_HYPRE(HYPRE_IJVectorAssemble(ij_sol));
        GKO_CHECK_HYPRE(
            HYPRE_IJVectorGetObject(ij_sol, reinterpret_cast<void**>(&par_sol)));

        // Tell HYPRE not to free the vector data — we manage the data
        // pointers ourselves (swapping between our buffers and external
        // Ginkgo data for zero-copy solves). Without this, destroy()
        // would call hypre_TFree on our std::vector memory.
        auto rhs_local = hypre_ParVectorLocalVector(par_rhs);
        auto sol_local = hypre_ParVectorLocalVector(par_sol);
        hypre_SeqVectorSetDataOwner(rhs_local, 0);
        hypre_SeqVectorSetDataOwner(sol_local, 0);
        // Free the HYPRE-allocated data and replace with our buffers
        hypre_TFree(hypre_VectorData(rhs_local), HYPRE_MEMORY_HOST);
        hypre_TFree(hypre_VectorData(sol_local), HYPRE_MEMORY_HOST);
        hypre_VectorData(rhs_local) = rhs_buffer.data();
        hypre_VectorData(sol_local) = sol_buffer.data();
    }

    /**
     * Point the hypre rhs/sol vectors at external data (zero-copy), or
     * copy into the internal buffers when direct viewing is not possible.
     * Returns whether the view was zero-copy (affects whether we need to
     * copy results back).
     */
    template <typename VT>
    bool set_vectors(const matrix::Dense<VT>* b, matrix::Dense<VT>* x,
                     size_type col)
    {
        const auto n = static_cast<size_type>(num_rows);
        // Zero-copy is possible when VT == double (matches HYPRE_Real)
        // and the column data is contiguous (stride == num_cols for
        // single-column, or more generally we check stride).
        constexpr bool is_double = std::is_same<VT, double>::value;
        const bool contiguous_b = (b->get_stride() == b->get_size()[1]);
        const bool contiguous_x = (x->get_stride() == x->get_size()[1]);

        if (is_double && contiguous_b && contiguous_x && b->get_size()[1] > 0) {
            // Direct view: point hypre at the Ginkgo data
            auto* b_col_ptr =
                const_cast<HYPRE_Real*>(reinterpret_cast<const HYPRE_Real*>(
                    b->get_const_values() + col));
            auto* x_col_ptr =
                reinterpret_cast<HYPRE_Real*>(x->get_values() + col);

            // For multi-column, column data has stride between rows
            // so direct view only works for single column with stride 1
            if (b->get_size()[1] == 1) {
                vector_data_ref(par_rhs) = b_col_ptr;
                vector_data_ref(par_sol) = x_col_ptr;
                return true;
            }
        }

        // Fallback: copy into internal buffers
        for (size_type i = 0; i < n; ++i) {
            rhs_buffer[i] = static_cast<HYPRE_Real>(b->at(i, col));
        }
        std::memset(sol_buffer.data(), 0, n * sizeof(HYPRE_Real));
        vector_data_ref(par_rhs) = rhs_buffer.data();
        vector_data_ref(par_sol) = sol_buffer.data();
        return false;
    }

    /**
     * Copy results from internal buffers back to the Dense output.
     * Only needed when set_vectors returned false (non-zero-copy path).
     */
    template <typename VT>
    void copy_back(matrix::Dense<VT>* x, size_type col)
    {
        const auto n = static_cast<size_type>(num_rows);
        for (size_type i = 0; i < n; ++i) {
            x->at(i, col) = static_cast<VT>(sol_buffer[i]);
        }
    }

    /**
     * Restore the hypre vector data pointers to the owned buffers.
     * Must be called after a zero-copy solve so that the hypre vectors
     * don't hold dangling pointers to temporary Ginkgo data.
     */
    void restore_owned_buffers()
    {
        vector_data_ref(par_rhs) = rhs_buffer.data();
        vector_data_ref(par_sol) = sol_buffer.data();
    }
};


template <typename ValueType>
HypreBoomerAmg<ValueType>::HypreBoomerAmg(std::shared_ptr<const Executor> exec)
    : EnableLinOp<HypreBoomerAmg>(std::move(exec))
{}


template <typename ValueType>
HypreBoomerAmg<ValueType>::HypreBoomerAmg(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<HypreBoomerAmg>(factory->get_executor(),
                                  system_matrix->get_size()),
      parameters_{factory->get_parameters()},
      state_(std::make_unique<hypre_state>())
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    ensure_hypre_init();

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // Convert to CSR<double, int32> on host
    auto host_csr = copy_and_convert_to<matrix::Csr<double, int32>>(
        host_exec, system_matrix);
    system_matrix_ = std::move(host_csr);

    setup_hypre();
}


template <typename ValueType>
void HypreBoomerAmg<ValueType>::setup_hypre()
{
    const auto n = static_cast<HYPRE_Int>(system_matrix_->get_size()[0]);

    state_->create_matrix(n, system_matrix_);
    state_->create_vectors(n);

    // Create and configure BoomerAMG
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGCreate(&state_->amg_solver));

    // Use as preconditioner: single cycle, zero tolerance
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetMaxIter(state_->amg_solver, 1));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetTol(state_->amg_solver, 0.0));

    // User-configurable parameters
    GKO_CHECK_HYPRE(
        HYPRE_BoomerAMGSetCycleType(state_->amg_solver, parameters_.cycle_type));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetCoarsenType(state_->amg_solver,
                                                  parameters_.coarsening_type));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetStrongThreshold(
        state_->amg_solver, parameters_.strength_threshold));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetRelaxType(state_->amg_solver,
                                                parameters_.smoother_type));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetNumSweeps(state_->amg_solver,
                                                parameters_.num_sweeps));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetInterpType(
        state_->amg_solver, parameters_.interpolation_type));
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetMaxLevels(state_->amg_solver,
                                                parameters_.max_levels));

    // Suppress output
    // TEMP DIAGNOSTIC: print level 1 dumps the BoomerAMG setup hierarchy and the
    // full effective parameter list (incl. knobs not set here, at hypre
    // defaults) once per setup. Revert to 0 when done comparing.
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetPrintLevel(state_->amg_solver, 1));

    // Run setup
    GKO_CHECK_HYPRE(HYPRE_BoomerAMGSetup(state_->amg_solver, state_->par_matrix,
                                         state_->par_rhs, state_->par_sol));
}


template <typename ValueType>
HypreBoomerAmg<ValueType>::~HypreBoomerAmg() = default;


template <typename ValueType>
HypreBoomerAmg<ValueType>::HypreBoomerAmg(const HypreBoomerAmg& other)
    : EnableLinOp<HypreBoomerAmg>(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
HypreBoomerAmg<ValueType>::HypreBoomerAmg(HypreBoomerAmg&& other) noexcept
    : EnableLinOp<HypreBoomerAmg>(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
HypreBoomerAmg<ValueType>& HypreBoomerAmg<ValueType>::operator=(
    const HypreBoomerAmg& other)
{
    if (this != &other) {
        EnableLinOp<HypreBoomerAmg>::operator=(other);
        parameters_ = other.parameters_;
        system_matrix_ = other.system_matrix_;
        if (other.state_ && other.state_->amg_solver) {
            ensure_hypre_init();
            state_ = std::make_unique<hypre_state>();
            setup_hypre();
        } else {
            state_.reset();
        }
    }
    return *this;
}


template <typename ValueType>
HypreBoomerAmg<ValueType>& HypreBoomerAmg<ValueType>::operator=(
    HypreBoomerAmg&& other) noexcept
{
    if (this != &other) {
        EnableLinOp<HypreBoomerAmg>::operator=(std::move(other));
        parameters_ = std::exchange(other.parameters_, {});
        system_matrix_ = std::move(other.system_matrix_);
        state_ = std::move(other.state_);
    }
    return *this;
}


template <typename ValueType>
void HypreBoomerAmg<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch<ValueType>(
        [this](auto dense_b, auto dense_x) {
            const auto exec = this->get_executor();
            const auto host_exec = exec->get_master();
            const auto size = dense_b->get_size();

            // Hypre runs on the host. If the operands already live on the
            // host we operate on them in place (which lets set_vectors take
            // its zero-copy path). Otherwise we stage them through the
            // persistent host buffers, which are reused across applies so we
            // don't reallocate on every call.
            const matrix::Dense<ValueType>* host_b;
            matrix::Dense<ValueType>* host_x;
            const bool needs_host_copy = host_exec != exec;
            if (needs_host_copy) {
                host_buffer_.init(host_exec, size);
                host_x_buffer_.init(host_exec, size);
                host_buffer_->copy_from(dense_b);
                host_b = host_buffer_.get();
                host_x = host_x_buffer_.get();
            } else {
                host_b = dense_b;
                host_x = dense_x;
            }

            for (size_type col = 0; col < size[1]; ++col) {
                bool zero_copy = state_->set_vectors(host_b, host_x, col);

                if (zero_copy) {
                    // set_vectors pointed hypre straight at host_x; zero the
                    // column to give the expected zero initial guess.
                    for (HYPRE_Int i = 0; i < state_->num_rows; ++i) {
                        host_x->at(i, col) = zero<ValueType>();
                    }
                }
                // In the non-zero-copy path set_vectors already zeroed the
                // internal solution buffer, so no initial guess is needed.

                GKO_CHECK_HYPRE_SOLVE(HYPRE_BoomerAMGSolve(
                    state_->amg_solver, state_->par_matrix, state_->par_rhs,
                    state_->par_sol));

                if (!zero_copy) {
                    state_->copy_back(host_x, col);
                }
            }

            // Drop hypre's pointers into host_b/host_x before those buffers
            // are reused on the next apply or destroyed.
            state_->restore_owned_buffers();

            if (needs_host_copy) {
                dense_x->copy_from(host_x);
            }
        },
        b, x);
}


template <typename ValueType>
void HypreBoomerAmg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    precision_dispatch<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_HYPRE_BOOMERAMG(ValueType) class HypreBoomerAmg<ValueType>
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_HYPRE_BOOMERAMG);


}  // namespace solver
}  // namespace experimental
}  // namespace gko
