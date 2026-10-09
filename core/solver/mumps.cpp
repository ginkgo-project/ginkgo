// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/mumps.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#endif


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

    // COO format storage for MUMPS (1-indexed, global indices in the
    // distributed case)
    std::vector<int> irn;
    std::vector<int> jcn;
    std::vector<ValueType> a;

    // Communicator MUMPS runs on: MPI_COMM_SELF for a sequential solve, the
    // ranks owning rows for a distributed one (MPI_COMM_NULL on the others).
    MPI_Comm comm = MPI_COMM_SELF;
    bool owns_comm = false;
    bool distributed = false;

    // Centralized right-hand side and solution: every rank sends its local
    // rows to the MUMPS host (rank 0 of comm) and receives them back.
    int n_local = 0;
    // On the host only: the global (0-based) row of every gathered entry, in
    // gather order, and the number of entries per rank of comm.
    std::vector<int> host_rows;
    std::vector<int> host_counts, host_offsets;
    std::vector<ValueType> host_rhs;  // n x nrhs, column-major
    std::vector<ValueType> gather_buf;
    std::vector<ValueType> local_buf;

    mumps_state() = default;

    ~mumps_state()
    {
        finalize();
        if (owns_comm && comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }

    mumps_state(const mumps_state&) = delete;
    mumps_state& operator=(const mumps_state&) = delete;
    mumps_state(mumps_state&&) = delete;
    mumps_state& operator=(mumps_state&&) = delete;

    void initialize(bool symmetric)
    {
        if (initialized) {
            finalize();
        }
        // NOTE: MUMPS is Fortran-based and requires the Fortran MPI
        // runtime (libmpifort) and ScaLAPACK to be linked. Without
        // them, MPI_Comm_c2f handles won't resolve correctly in the
        // Fortran layer, causing "Instance Error 1".
        std::memset(&data, 0, sizeof(data));
        data.par = 1;                  // host participates in factorization
        data.sym = symmetric ? 2 : 0;  // 0 = unsymmetric, 2 = symmetric
        data.comm_fortran = MPI_Comm_c2f(comm);
        data.job = mumps_job_init;
        traits::mumps_c(&data);
        initialized = true;

        // Suppress all MUMPS output
        data.icntl[0] = -1;   // no error messages
        data.icntl[1] = -1;   // no diagnostic output
        data.icntl[2] = -1;   // no global info
        data.icntl[3] = 0;    // verbosity level off
        data.icntl[13] = 50;  // ICNTL(14): % memory relaxation (default 20)
        if (distributed) {
            // ICNTL(18): distributed assembled matrix. The right-hand side
            // and solution stay centralized on the host (ICNTL(20) = 0,
            // ICNTL(21) = 0).
            data.icntl[17] = 3;
        }
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
        if (distributed) {
            data.nnz_loc = nnz;
            data.irn_loc = irn_ptr;
            data.jcn_loc = jcn_ptr;
            data.a_loc = mumps_value_ptr(a_ptr);
        } else {
            data.nnz = nnz;
            data.irn = irn_ptr;
            data.jcn = jcn_ptr;
            data.a = mumps_value_ptr(a_ptr);
        }
    }

    // INFOG is global, so all ranks of comm throw together.
    void run(int job, const char* phase)
    {
        data.job = job;
        traits::mumps_c(&data);
        GKO_THROW_IF_INVALID(data.infog[0] >= 0,
                             std::string("MUMPS ") + phase +
                                 " failed with INFOG(1) = " +
                                 std::to_string(data.infog[0]));
    }

    void analyze() { run(mumps_job_analyze, "analysis"); }

    void factorize() { run(mumps_job_factorize, "factorization"); }

    /**
     * Solves the system. rhs_ptr must point to a column-major array of
     * size (ldrhs, nrhs) where ldrhs >= n.
     */
    void solve(int nrhs, ValueType* rhs_ptr, int ldrhs)
    {
        data.nrhs = nrhs;
        data.lrhs = ldrhs;
        data.rhs = mumps_value_ptr(rhs_ptr);
        run(mumps_job_solve, "solve");
    }

    /**
     * Solves with the right-hand side distributed like the matrix rows:
     * local holds nrhs column-major columns of n_local entries, and is
     * overwritten with the solution. The columns are gathered to the host,
     * solved there and scattered back.
     */
    void solve_centralized(int nrhs, ValueType* local)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
        const auto type =
            experimental::mpi::type_impl<ValueType>::get_type();
        const auto n = data.n;
        if (rank == 0) {
            host_rhs.resize(static_cast<size_type>(n) * nrhs);
            gather_buf.resize(host_rows.size());
        }
        for (int j = 0; j < nrhs; j++) {
            MPI_Gatherv(local + static_cast<size_type>(j) * n_local, n_local,
                        type, gather_buf.data(), host_counts.data(),
                        host_offsets.data(), type, 0, comm);
            if (rank == 0) {
                auto col = host_rhs.data() + static_cast<size_type>(j) * n;
                for (size_type k = 0; k < host_rows.size(); k++) {
                    col[host_rows[k]] = gather_buf[k];
                }
            }
        }
        // The rhs is only read on the host.
        solve(nrhs, rank == 0 ? host_rhs.data() : nullptr, n);
        for (int j = 0; j < nrhs; j++) {
            if (rank == 0) {
                auto col = host_rhs.data() + static_cast<size_type>(j) * n;
                for (size_type k = 0; k < host_rows.size(); k++) {
                    gather_buf[k] = col[host_rows[k]];
                }
            }
            MPI_Scatterv(gather_buf.data(), host_counts.data(),
                         host_offsets.data(), type,
                         local + static_cast<size_type>(j) * n_local, n_local,
                         type, 0, comm);
        }
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


#if GINKGO_BUILD_MPI


namespace {


/**
 * Sets up a distributed factorization if system_matrix is a
 * distributed::Matrix<ValueType, IndexType, GlobalIndexType>: collects the
 * local and non-local block as COO in global 1-based indices, splits off the
 * communicator of the ranks owning rows, and collects the global indices of
 * every rank's rows on its host for gathering the right-hand side. Returns
 * false if the matrix has a different type.
 */
template <typename GlobalIndexType, typename ValueType, typename IndexType,
          typename State>
bool setup_distributed(const LinOp* system_matrix, bool lower_only,
                       State& state)
{
    using dist_mtx = experimental::distributed::Matrix<ValueType, IndexType,
                                                       GlobalIndexType>;
    using csr = matrix::Csr<ValueType, IndexType>;
    using experimental::distributed::index_space;
    auto dist = dynamic_cast<const dist_mtx*>(system_matrix);
    if (!dist) {
        return false;
    }
    const auto exec = dist->get_executor();
    const auto host_exec = exec->get_master();
    const auto& imap = dist->get_index_map();
    GKO_THROW_IF_INVALID(
        dist->get_size()[0] <=
            static_cast<size_type>(std::numeric_limits<int>::max()),
        "distributed MUMPS needs a global size that fits into int");

    auto global_idxs = [&](size_type n, index_space space) {
        array<IndexType> local{exec, n};
        if (n > 0) {
            std::iota(local.get_data(), local.get_data() + n, IndexType{});
        }
        auto global = imap.map_to_global(local, space);
        global.set_executor(host_exec);
        return global;
    };
    // The local block's rows and columns share the local numbering, since
    // the row and column partitions are the same.
    const auto n_local = imap.get_local_size();
    const auto local_global = global_idxs(n_local, index_space::local);
    const auto non_local_global =
        global_idxs(imap.get_non_local_size(), index_space::non_local);
    const auto rows = local_global.get_const_data();

    state.irn.clear();
    state.jcn.clear();
    state.a.clear();
    auto add_block = [&](std::shared_ptr<const LinOp> block,
                         const GlobalIndexType* cols) {
        auto host_block = copy_and_convert_to<csr>(host_exec, block);
        const auto row_ptrs = host_block->get_const_row_ptrs();
        const auto col_idxs = host_block->get_const_col_idxs();
        const auto values = host_block->get_const_values();
        for (size_type row = 0; row < host_block->get_size()[0]; row++) {
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto grow = rows[row];
                const auto gcol = cols[col_idxs[idx]];
                if (lower_only && gcol > grow) {
                    continue;
                }
                state.irn.push_back(static_cast<int>(grow) + 1);
                state.jcn.push_back(static_cast<int>(gcol) + 1);
                state.a.push_back(values[idx]);
            }
        }
    };
    add_block(dist->get_local_matrix(), rows);
    add_block(dist->get_non_local_matrix(), non_local_global.get_const_data());

    state.distributed = true;
    state.n_local = static_cast<int>(n_local);

    // Only the ranks owning rows take part.
    const auto comm = dist->get_communicator();
    if (state.owns_comm && state.comm != MPI_COMM_NULL) {
        MPI_Comm_free(&state.comm);
    }
    MPI_Comm_split(comm.get(), n_local > 0 ? 0 : MPI_UNDEFINED, comm.rank(),
                   &state.comm);
    state.owns_comm = true;
    if (state.comm == MPI_COMM_NULL) {
        return true;
    }
    int sub_size, sub_rank;
    MPI_Comm_size(state.comm, &sub_size);
    MPI_Comm_rank(state.comm, &sub_rank);
    std::vector<int> local_rows(rows, rows + n_local);
    if (sub_rank == 0) {
        state.host_counts.resize(sub_size);
    }
    MPI_Gather(&state.n_local, 1, MPI_INT, state.host_counts.data(), 1,
               MPI_INT, 0, state.comm);
    if (sub_rank == 0) {
        state.host_offsets.assign(sub_size + 1, 0);
        std::partial_sum(state.host_counts.begin(), state.host_counts.end(),
                         state.host_offsets.begin() + 1);
        state.host_rows.resize(state.host_offsets.back());
        // Every global row is owned by exactly one rank.
        GKO_THROW_IF_INVALID(
            state.host_rows.size() == dist->get_size()[0],
            "distributed MUMPS needs every row to be owned by one rank");
    }
    MPI_Gatherv(local_rows.data(), state.n_local, MPI_INT,
                state.host_rows.data(), state.host_counts.data(),
                state.host_offsets.data(), MPI_INT, 0, state.comm);
    return true;
}


}  // namespace


#endif  // GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
void Mumps<ValueType, IndexType>::generate()
{
    const auto host_exec = this->get_executor()->get_master();
    state_ = std::make_unique<mumps_state>();

#if GINKGO_BUILD_MPI
    bool distributed = false;
    if constexpr (std::is_same<IndexType, int32>::value) {
        distributed = setup_distributed<int32, ValueType, IndexType>(
            system_matrix_.get(), parameters_.symmetric, *state_);
    }
    if (!distributed) {
        distributed = setup_distributed<int64, ValueType, IndexType>(
            system_matrix_.get(), parameters_.symmetric, *state_);
    }
    if (distributed) {
        if (state_->comm != MPI_COMM_NULL) {
            state_->initialize(parameters_.symmetric);
            state_->set_ordering(parameters_.ordering);
            state_->set_matrix(static_cast<int>(system_matrix_->get_size()[0]),
                               static_cast<int>(state_->irn.size()),
                               state_->irn.data(), state_->jcn.data(),
                               state_->a.data());
            state_->analyze();
            state_->factorize();
        }
        return;
    }
#endif

    // Convert to CSR on host
    if (!dynamic_cast<const matrix_type*>(system_matrix_.get()) ||
        system_matrix_->get_executor() != host_exec) {
        system_matrix_ = copy_and_convert_to<matrix_type>(host_exec,
                                                          system_matrix_);
    }
    const auto csr = as<matrix_type>(system_matrix_.get());
    const auto num_rows = static_cast<int>(csr->get_size()[0]);
    // An empty local problem (e.g. a rank without coarse dofs) needs no
    // factorization, and its apply has nothing to do.
    if (num_rows == 0) {
        return;
    }

    csr_to_coo_1based(num_rows, csr->get_const_row_ptrs(),
                      csr->get_const_col_idxs(), csr->get_const_values(),
                      state_->irn, state_->jcn, state_->a,
                      parameters_.symmetric);

    const auto nnz = static_cast<int>(state_->irn.size());

    // Use MPI_COMM_SELF so each rank solves independently.
    state_->initialize(parameters_.symmetric);
    state_->set_ordering(parameters_.ordering);
    state_->set_matrix(num_rows, nnz, state_->irn.data(), state_->jcn.data(),
                       state_->a.data());
    state_->analyze();
    state_->factorize();
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
      system_matrix_{std::move(system_matrix)}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);
    this->generate();
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
        // Re-factorize from stored matrix (collectively, if it is
        // distributed)
        if (other.state_) {
            this->generate();
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
void Mumps<ValueType, IndexType>::apply_distributed(const LinOp* b,
                                                    LinOp* x) const
{
#if GINKGO_BUILD_MPI
    using vec = experimental::distributed::Vector<ValueType>;
    const auto host_exec = this->get_executor()->get_master();
    auto dist_b = dynamic_cast<const vec*>(b);
    auto dist_x = dynamic_cast<vec*>(x);
    GKO_THROW_IF_INVALID(dist_b && dist_x,
                         "distributed MUMPS needs distributed vectors of "
                         "its value type");
    auto& state = *state_;
    if (state.comm == MPI_COMM_NULL) {
        return;  // no rows here
    }
    auto host_b = make_temporary_clone(host_exec, dist_b->get_local_vector());
    const auto n_local = static_cast<size_type>(state.n_local);
    const auto nrhs = host_b->get_size()[1];
    state.local_buf.resize(n_local * nrhs);
    for (size_type j = 0; j < nrhs; j++) {
        for (size_type i = 0; i < n_local; i++) {
            state.local_buf[j * n_local + i] = host_b->at(i, j);
        }
    }
    state.solve_centralized(static_cast<int>(nrhs), state.local_buf.data());
    const auto local_size = dist_x->get_local_vector()->get_size();
    const auto local_stride = dist_x->get_local_vector()->get_stride();
    auto x_vals =
        matrix::Dense<ValueType>::create(host_exec, local_size, local_stride);
    for (size_type j = 0; j < nrhs; j++) {
        for (size_type i = 0; i < n_local; i++) {
            x_vals->at(i, j) = state.local_buf[j * n_local + i];
        }
    }
    if (local_size[0] > 0) {
        dist_x->get_executor()->copy_from(
            host_exec, (local_size[0] - 1) * local_stride + local_size[1],
            x_vals->get_const_values(), dist_x->get_local_values());
    }
#else
    GKO_NOT_SUPPORTED(b);
#endif
}


template <typename ValueType, typename IndexType>
void Mumps<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (state_ && state_->distributed) {
        this->apply_distributed(b, x);
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            const auto host_exec = this->get_executor()->get_master();
            const auto size = dense_b->get_size();
            const auto num_rows = static_cast<int>(size[0]);
            if (num_rows == 0) {
                return;
            }

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
    if (state_ && state_->distributed) {
#if GINKGO_BUILD_MPI
        using vec = experimental::distributed::Vector<ValueType>;
        auto x_clone = x->clone();
        this->apply_impl(b, x_clone.get());
        as<vec>(x)->scale(beta);
        as<vec>(x)->add_scaled(alpha, x_clone);
#endif
        return;
    }
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
