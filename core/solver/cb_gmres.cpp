// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cb_gmres.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/extended_float.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/solver/cb_gmres_accessor.hpp"
#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace solver {
namespace cb_gmres {
namespace {


GKO_REGISTER_OPERATION(initialize, cb_gmres::initialize);
GKO_REGISTER_OPERATION(restart, cb_gmres::restart);
GKO_REGISTER_OPERATION(arnoldi, cb_gmres::arnoldi);
GKO_REGISTER_OPERATION(gram_column, cb_gmres::gram_column);
GKO_REGISTER_OPERATION(solve_krylov, cb_gmres::solve_krylov);


// Optional per-cycle instrumentation, enabled by setting
// GINKGO_CB_GMRES_CYCLE_LOG to the path of a file to append to, or to "-" for
// standard error. Unset, nothing is recorded and nothing is written. Read on
// every call rather than cached in a static: one process may write several
// logs, a driver that solves one matrix under many configurations redirecting
// the variable between solves.
const char* cycle_log_path()
{
    return std::getenv("GINKGO_CB_GMRES_CYCLE_LOG");
}


// Identifies one call of the solver, so that the records of several solves
// appended to the same file can be told apart.
int next_solve_index()
{
    static std::atomic<int> counter{0};
    return counter++;
}


// One row of the per-cycle log. `reason` is the criterion that ended the
// cycle: the restart tolerance, the cap on the basis, a reset from the
// convergence detection, or the end of the solve.
struct cycle_record {
    int solve;
    int index;
    int start_iter;
    int length;
    const char* reason;
    double res_start;
    double res_end;
    size_type reorth;
    // psi_j = ||V_j^T w||_2 / h_{j+1,j} over the steps of this cycle that
    // took a second orthogonalization pass, under GKO_CB_GMRES_PSI. psi_n
    // counts the contributing steps; the fields are written empty when it is
    // zero.
    double psi_max;
    double psi_mean;
    int psi_n;
    // From the same two vectors: chi_j = ||h_{1:j,j}|| / h_{j+1,j}, the
    // cancellation ratio, and theta_lb_j = ||V_j^T w|| / ||h_{1:j,j}||, a
    // lower bound on the loss of orthogonality. Both share psi_n.
    double theta_lb_max;
    double theta_lb_mean;
    double chi_max;
    double chi_mean;
    // ||v_j||_inf / ||v_j||_2 over the basis vectors written in this cycle,
    // read from the accessor's scale, so only for a scaled accessor and a
    // single right-hand side. vinf_count is zero when it was not collected.
    double vinf_max;
    double vinf_mean;
    int vinf_count;
    // The loss of orthogonality measured rather than bounded, under
    // GKO_CB_GMRES_GRAM. With S = I - Vbar^T Vbar over the basis the cycle
    // ended with, theta2 = ||S||_2, theta_diag = max_j |1 - ||v_j||^2| is its
    // diagonal part and theta_off the Frobenius norm of the rest. gram_n is
    // the number of stored vectors it was taken over.
    double theta2;
    double theta_fro;
    double theta_diag;
    double theta_off;
    int gram_n;
    // ||v_j||_inf / ||v_j||_2 taken from the Arnoldi step instead of the
    // accessor's scale, so that it exists at the floating-point formats too.
    double loc_max;
    double loc_mean;
    int loc_n;
};


// Appends the records of one solve, once, after the solve.
void write_cycle_log(const std::vector<cycle_record>& records)
{
    const char* path = cycle_log_path();
    if (path == nullptr || records.empty()) {
        return;
    }
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> guard{log_mutex};
    std::ostringstream os;
    for (const auto& r : records) {
        os << r.solve << ',' << r.index << ',' << r.start_iter << ','
           << r.length << ',' << r.reason << ',' << r.res_start << ','
           << r.res_end << ',' << r.reorth << ',';
        if (r.psi_n > 0) {
            os << r.psi_max << ',' << r.psi_mean << ',' << r.psi_n << ','
               << r.theta_lb_max << ',' << r.theta_lb_mean << ',' << r.chi_max
               << ',' << r.chi_mean << ',';
        } else {
            os << ",,0,,,,,";
        }
        if (r.vinf_count > 0) {
            os << r.vinf_max << ',' << r.vinf_mean << ',' << r.vinf_count;
        } else {
            os << ",,0";
        }
        os << ',';
        if (r.gram_n > 0) {
            os << r.theta2 << ',' << r.theta_fro << ',' << r.theta_diag << ','
               << r.theta_off << ',' << r.gram_n;
        } else {
            os << ",,,,0";
        }
        os << ',';
        if (r.loc_n > 0) {
            os << r.loc_max << ',' << r.loc_mean << ',' << r.loc_n;
        } else {
            os << ",,0";
        }
        os << '\n';
    }
    if (std::string{path} == "-") {
        std::cerr << os.str();
        std::cerr.flush();
    } else {
        // Header once per file, not once per process: opening for append
        // leaves the put pointer at the end, so a zero offset means the file
        // is new or empty.
        std::ofstream out{path, std::ios::app};
        if (out) {
            if (out.tellp() == std::streampos{0}) {
                out << "solve,cycle,start_iter,length,reason,res_start,"
                       "res_end,reorth,psi_max,psi_mean,psi_n,"
                       "theta_lb_max,theta_lb_mean,chi_max,chi_mean,"
                       "vinf_max,vinf_mean,vinf_n,"
                       "theta2,theta_fro,theta_diag,theta_off,gram_n,"
                       "loc_max,loc_mean,loc_n\n";
            }
            out << os.str();
        }
    }
}


// The spectral norm of a small symmetric matrix, by cyclic Jacobi rotations on
// a copy of it: for a symmetric matrix the largest eigenvalue in modulus is the
// 2-norm. The matrix here is (j+1) x (j+1) with j the basis size of a cycle.
double symmetric_spectral_norm(std::vector<double> a, int n)
{
    if (n <= 0) {
        return 0.0;
    }
    const auto at = [&](int i, int j) -> double& { return a[i * n + j]; };
    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scale += at(i, j) * at(i, j);
        }
    }
    const double tol = 1e-28 * (scale > 0.0 ? scale : 1.0);
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                off += at(i, j) * at(i, j);
            }
        }
        if (off <= tol) {
            break;
        }
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = at(p, q);
                if (apq == 0.0) {
                    continue;
                }
                const double theta = (at(q, q) - at(p, p)) / (2.0 * apq);
                const double t =
                    (theta >= 0.0 ? 1.0 : -1.0) /
                    (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double sn = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = at(k, p);
                    const double akq = at(k, q);
                    at(k, p) = c * akp - sn * akq;
                    at(k, q) = sn * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = at(p, k);
                    const double aqk = at(q, k);
                    at(p, k) = c * apk - sn * aqk;
                    at(q, k) = sn * apk + c * aqk;
                }
            }
        }
    }
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        m = std::max(m, std::abs(at(i, i)));
    }
    return m;
}


}  // anonymous namespace
}  // namespace cb_gmres


template <typename T>
struct to_integer_impl {
    using type = T;
};

template <>
struct to_integer_impl<double> {
    using type = int64;
};

template <>
struct to_integer_impl<float> {
    using type = int32;
};

template <>
struct to_integer_impl<half> {
    using type = int16;
};

template <typename T>
using to_integer = typename to_integer_impl<T>::type;


template <typename T, typename Skip>
using reduce_precision_skip =
    typename std::conditional_t<std::is_same<reduce_precision<T>, Skip>::value,
                                T, reduce_precision<T>>;


namespace detail {


template <typename T, typename Skip, int count>
struct reduce_precision_skip_count_impl {
    static_assert(count > 0,
                  "The count variable must be larger or equal to zero.");
    using type = typename reduce_precision_skip_count_impl<
        reduce_precision_skip<T, Skip>, Skip, count - 1>::type;
};

template <typename T, typename Skip>
struct reduce_precision_skip_count_impl<T, Skip, 0> {
    using type = T;
};


}  // namespace detail


template <typename T, typename Skip, int count>
using reduce_precision_skip_count =
    typename detail::reduce_precision_skip_count_impl<T, Skip, count>::type;


template <typename T, int count>
using reduce_precision_count =
    typename detail::reduce_precision_skip_count_impl<T, void, count>::type;


template <typename ValueType>
struct helper {
    template <typename Callable>
    static void call(Callable callable,
                     gko::solver::cb_gmres::storage_precision st)
    {
        switch (st) {
        case cb_gmres::storage_precision::reduce1:
            callable(reduce_precision_count<ValueType, 1>{});
            break;
        case cb_gmres::storage_precision::reduce2:
            callable(reduce_precision_count<ValueType, 2>{});
            break;
        case cb_gmres::storage_precision::integer:
            callable(to_integer<ValueType>{});
            break;
        case cb_gmres::storage_precision::ireduce1:
            callable(to_integer<reduce_precision_count<ValueType, 1>>{});
            break;
        case cb_gmres::storage_precision::ireduce2:
            callable(to_integer<reduce_precision_count<ValueType, 2>>{});
            break;
        default:
            callable(ValueType{});
        }
    }
};


// helper for complex numbers
template <typename T>
struct helper<std::complex<T>> {
    using ValueType = std::complex<T>;
    using skip_type = std::complex<half>;

    template <typename Callable>
    static void call(Callable callable,
                     gko::solver::cb_gmres::storage_precision st)
    {
        switch (st) {
        case cb_gmres::storage_precision::reduce1:
            callable(reduce_precision_skip_count<ValueType, skip_type, 1>{});
            break;
        case cb_gmres::storage_precision::reduce2:
            callable(reduce_precision_skip_count<ValueType, skip_type, 2>{});
            break;
        case cb_gmres::storage_precision::integer:
        case cb_gmres::storage_precision::ireduce1:
        case cb_gmres::storage_precision::ireduce2:
            GKO_NOT_SUPPORTED(st);
            break;
        default:
            callable(ValueType{});
        }
    }
};


template <typename ValueType>
typename CbGmres<ValueType>::parameters_type CbGmres<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::CbGmres<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);
    if (auto& obj = config_check.get("krylov_dim")) {
        params.with_krylov_dim(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("restart_ratio")) {
        params.with_restart_ratio(
            gko::config::get_value<remove_complex<ValueType>>(obj));
    }
    if (auto& obj = config_check.get("max_reorth_steps")) {
        params.with_max_reorth_steps(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("storage_precision")) {
        auto get_storage_precision = [](const std::string& str) {
            using gko::solver::cb_gmres::storage_precision;
            if (str == "keep") {
                return storage_precision::keep;
            } else if (str == "reduce1") {
                return storage_precision::reduce1;
            } else if (str == "reduce2") {
                return storage_precision::reduce2;
            } else if (str == "integer") {
                return storage_precision::integer;
            } else if (str == "ireduce1") {
                return storage_precision::ireduce1;
            } else if (str == "ireduce2") {
                return storage_precision::ireduce2;
            }
            GKO_INVALID_CONFIG_VALUE("storage_precision", str);
        };
        params.with_storage_precision(get_storage_precision(obj.get_string()));
    }

    return params;
}


template <typename ValueType>
void CbGmres<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void CbGmres<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    // Current workaround to get a lambda with a template argument (only
    // the type of `value` matters, the content does not)
    auto apply_templated = [&](auto value) {
        using storage_type = decltype(value);

        using Vector = matrix::Dense<ValueType>;
        using VectorNorms = matrix::Dense<remove_complex<ValueType>>;
        using Range3dHelper =
            gko::cb_gmres::Range3dHelper<ValueType, storage_type>;


        constexpr uint8 RelativeStoppingId{1};

        auto exec = this->get_executor();

        auto one_op = initialize<Vector>({one<ValueType>()}, exec);
        auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

        const auto num_rows = this->get_size()[0];
        const auto num_rhs = dense_b->get_size()[1];
        const auto krylov_dim = this->get_krylov_dim();
        auto residual = Vector::create_with_config_of(dense_b);
        /* The dimensions {x, y, z} explained for the krylov_bases:
         * - x: selects the krylov vector (which has krylov_dim + 1 vectors)
         * - y: selects the (row-)element of said krylov vector
         * - z: selects which column-element of said krylov vector should be
         *      used
         */
        const dim<3> krylov_bases_dim{krylov_dim + 1, num_rows, num_rhs};
        Range3dHelper helper(exec, krylov_bases_dim);
        auto krylov_bases_range = helper.get_range();

        auto next_krylov_basis = Vector::create_with_config_of(dense_b);
        std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
            Vector::create_with_config_of(dense_b);
        auto hessenberg =
            Vector::create(exec, dim<2>{krylov_dim + 1, krylov_dim * num_rhs});
        auto buffer = Vector::create(exec, dim<2>{krylov_dim + 1, num_rhs});
        auto givens_sin = Vector::create(exec, dim<2>{krylov_dim, num_rhs});
        auto givens_cos = Vector::create(exec, dim<2>{krylov_dim, num_rhs});
        auto residual_norm_collection =
            Vector::create(exec, dim<2>{krylov_dim + 1, num_rhs});
        auto residual_norm = VectorNorms::create(exec, dim<2>{1, num_rhs});
        // 1st row of arnoldi_norm: == eta * norm2(old_next_krylov_basis)
        //                          with eta == 1 / sqrt(2)
        //                          (computed right before updating
        //                          next_krylov_basis)
        // 2nd row of arnoldi_norm: The actual arnoldi norm
        //                          == norm2(next_krylov_basis)
        // 3rd row of arnoldi_norm: the infinity norm of next_krylov_basis
        //                          (ONLY when using a scalar accessor)
        auto arnoldi_norm = VectorNorms::create(exec, dim<2>{3, num_rhs});
        // One column of the Gram matrix of the stored basis, for the
        // diagnostic measurement of the loss of orthogonality below.
        auto gram_col = Vector::create(exec, dim<2>{krylov_dim + 1, num_rhs});
        array<size_type> final_iter_nums(this->get_executor(), num_rhs);
        auto y = Vector::create(exec, dim<2>{krylov_dim, num_rhs});

        bool one_changed{};
        array<char> reduction_tmp{this->get_executor()};
        array<stopping_status> stop_status(this->get_executor(), num_rhs);
        // reorth_status and num_reorth are both helper variables for GPU
        // implementations at the moment.
        // num_reorth := Number of vectors which require a re-orthogonalization
        // reorth_status := stopping status for the re-orthogonalization,
        //                  marking which RHS requires one, and which does not
        array<stopping_status> reorth_status(this->get_executor(), num_rhs);
        array<size_type> num_reorth(this->get_executor(), 1);

        // Initialization
        exec->run(cb_gmres::make_initialize(
            dense_b->get_const_device_view(), residual->get_device_view(),
            givens_sin->get_device_view(), givens_cos->get_device_view(),
            stop_status, krylov_dim));
        // residual = dense_b
        // givens_sin = givens_cos = 0
        this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
        // residual = residual - Ax

        exec->run(cb_gmres::make_restart(
            residual->get_const_device_view(), residual_norm->get_device_view(),
            residual_norm_collection->get_device_view(),
            arnoldi_norm->get_device_view(), krylov_bases_range,
            next_krylov_basis->get_device_view(), final_iter_nums,
            reduction_tmp, krylov_dim));
        // residual_norm = norm(residual)
        // residual_norm_collection = {residual_norm, 0, ..., 0}
        // krylov_bases(:, 1) = residual / residual_norm
        // next_krylov_basis = residual / residual_norm
        // final_iter_nums = {0, ..., 0}

        auto stop_criterion = this->get_stop_criterion_factory()->generate(
            this->get_system_matrix(),
            std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
            residual.get());

        int total_iter = -1;
        size_type restart_iter = 0;

        auto before_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);
        auto after_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);

        array<bool> stop_encountered_rhs(exec->get_master(), num_rhs);
        array<bool> fully_converged_rhs(exec->get_master(), num_rhs);
        array<stopping_status> host_stop_status(
            this->get_executor()->get_master(), stop_status);
        for (size_type i = 0; i < stop_encountered_rhs.get_size(); ++i) {
            stop_encountered_rhs.get_data()[i] = false;
            fully_converged_rhs.get_data()[i] = false;
        }

        // Ratio-based restart. When restart_ratio is positive, a cycle ends
        // once the correction has reduced the residual of the correction
        // system by that factor, for every right-hand side not already
        // converged. Zero disables it and restores the pure krylov_dim
        // restart.
        const auto restart_ratio = this->get_restart_ratio();
        const bool use_restart_ratio =
            restart_ratio > zero<remove_complex<ValueType>>();
        // Per-cycle instrumentation, off unless the variable is set.
        const bool log_cycles = cb_gmres::cycle_log_path() != nullptr;
        const bool need_cycle_norms = use_restart_ratio || log_cycles;
        // ||r_i||, the residual norm when the current cycle started, and the
        // current ||r_i - A d_i||, both kept on the host.
        auto cycle_start_norm =
            VectorNorms::create(exec->get_master(), dim<2>{1, num_rhs});
        auto host_residual_norm =
            VectorNorms::create(exec->get_master(), dim<2>{1, num_rhs});
        std::vector<cb_gmres::cycle_record> cycle_records;
        // ||v_j||_inf / ||v_j||_2, which the Arnoldi kernel already writes
        // into the per-vector scale of a scaled accessor. Restricted to a
        // single right-hand side, where that array is contiguous in the vector
        // index; a multi-rhs run records nothing rather than guessing at the
        // masked layout.
        const bool log_vinf =
            log_cycles && Range3dHelper::has_scale && num_rhs == 1;
        // write_scalar multiplies the ratio by this before storing it, so
        // dividing it out recovers the ratio. The branch has to be
        // compile-time: numeric_limits<>::max() does not convert to double for
        // a complex storage type.
        double scalar_correction = 1.0;
        if constexpr (std::is_integral<storage_type>::value) {
            scalar_correction =
                2.0 /
                static_cast<double>(std::numeric_limits<storage_type>::max());
        }
        // Allocated once at the full basis size. Holds ValueType, not its
        // real part, because that is what the accessor's scale array holds.
        array<ValueType> host_scale{exec->get_master(),
                                    log_vinf ? krylov_dim + 1 : 0};
        const int solve_index = log_cycles ? cb_gmres::next_solve_index() : 0;
        int cycle_index = 0;
        int cycle_start_iter = 0;
        const char* pending_reason = "cap";
        // Counts (right-hand side x pass) re-orthogonalizations, passed to
        // the kernel only when the log is on.
        size_type reorth_passes = 0;
        size_type cycle_start_reorth = 0;
        size_type* reorth_counter = log_cycles ? &reorth_passes : nullptr;
        // Off unless GKO_CB_GMRES_PSI is set: it copies the projection
        // coefficients of the second pass to the host once per step.
        const bool log_psi =
            log_cycles && std::getenv("GKO_CB_GMRES_PSI") != nullptr;
        kernels::cb_gmres_psi_stats psi_stats{};
        kernels::cb_gmres_psi_stats* psi_counter =
            log_psi ? &psi_stats : nullptr;
        // The exact loss of orthogonality, off unless GKO_CB_GMRES_GRAM is
        // set: it forms one column of the Gram matrix of the stored basis per
        // step and so traverses it once more.
        const bool log_gram = log_cycles && num_rhs == 1 &&
                              std::getenv("GKO_CB_GMRES_GRAM") != nullptr;
        const auto gram_ld = krylov_dim + 1;
        std::vector<double> gram(log_gram ? gram_ld * gram_ld : 0, 0.0);
        auto host_gram_col =
            log_gram ? Vector::create(exec->get_master(),
                                      dim<2>{krylov_dim + 1, num_rhs})
                     : nullptr;
        int gram_n = 0;
        // Forms column `index` of the Gram matrix of the stored basis and
        // keeps it, symmetrically, in `gram`.
        auto record_gram_column = [&](size_type index) {
            if (!log_gram || index >= gram_ld) {
                return;
            }
            exec->run(cb_gmres::make_gram_column(
                krylov_bases_range.get_accessor().to_const(),
                gram_col->get_device_view(), index));
            host_gram_col->copy_from(gram_col);
            for (size_type k = 0; k <= index; ++k) {
                const auto v =
                    static_cast<double>(real(host_gram_col->at(k, 0)));
                gram[k * gram_ld + index] = v;
                gram[index * gram_ld + k] = v;
            }
            gram_n = static_cast<int>(index + 1);
        };
        // Records the residual norm the cycle starts from, which the
        // tolerance is measured against.
        auto record_cycle_start = [&](const VectorNorms* res_norm) {
            if (need_cycle_norms) {
                cycle_start_norm->copy_from(res_norm);
            }
            // The first stored vector contributes its own norm, which the
            // later columns do not cover.
            record_gram_column(0);
        };
        // Reduces the per-vector scales of the current cycle to a maximum
        // and a sum. Only called when log_vinf holds, so the scale array
        // exists and num_rhs is one.
        auto collect_vinf = [&](int length, double& out_max, double& out_sum,
                                int& out_count) {
            // Compile-time guard: an unscaled helper has no get_scale at
            // all, and the body is instantiated for it regardless.
            if constexpr (Range3dHelper::has_scale) {
                // Vector 0 is the residual that opened the cycle, and a
                // cycle of length L writes vectors 1 through L.
                const auto wanted = std::min<size_type>(
                    static_cast<size_type>(length) + 1, krylov_dim + 1);
                if (wanted == 0) {
                    return;
                }
                auto& scale = helper.get_scale();
                exec->get_master()->copy_from(exec, wanted,
                                              scale.get_const_data(),
                                              host_scale.get_data());
                for (size_type j = 0; j < wanted; ++j) {
                    const auto ratio = static_cast<double>(abs(
                                           host_scale.get_const_data()[j])) /
                                       scalar_correction;
                    // A vector that was never written leaves the scale at its
                    // initial value of one, which is not a measurement.
                    if (!(ratio > 0.0) || ratio > 1.0) {
                        continue;
                    }
                    out_max = std::max(out_max, ratio);
                    out_sum += ratio;
                    ++out_count;
                }
            }
        };
        // Closes the current cycle in the log, with `reason` the criterion
        // that ended it and end_iter the last completed step.
        auto close_cycle = [&](const char* reason, int end_iter) {
            if (!log_cycles) {
                return;
            }
            host_residual_norm->copy_from(residual_norm);
            double res_end = 0.0;
            double res_start = 0.0;
            for (size_type i = 0; i < num_rhs; ++i) {
                res_end = std::max(
                    res_end, static_cast<double>(host_residual_norm->at(0, i)));
                res_start = std::max(
                    res_start, static_cast<double>(cycle_start_norm->at(0, i)));
            }
            // Indices are cycle-local: a cycle restarts the basis.
            double vinf_max = 0.0;
            double vinf_sum = 0.0;
            int vinf_count = 0;
            if (log_vinf) {
                collect_vinf(end_iter - cycle_start_iter, vinf_max, vinf_sum,
                             vinf_count);
            }
            // S = I - Vbar^T Vbar over the basis this cycle ended with,
            // split into its diagonal, which a write biased toward zero fills
            // by shortening every stored vector, and the rest.
            double theta2 = 0.0;
            double theta_fro = 0.0;
            double theta_diag = 0.0;
            double theta_off = 0.0;
            if (gram_n > 0) {
                std::vector<double> sm(static_cast<size_type>(gram_n) * gram_n);
                for (int r = 0; r < gram_n; ++r) {
                    for (int c = 0; c < gram_n; ++c) {
                        const double v =
                            (r == c ? 1.0 : 0.0) -
                            gram[static_cast<size_type>(r) * gram_ld + c];
                        sm[static_cast<size_type>(r) * gram_n + c] = v;
                        theta_fro += v * v;
                        if (r == c) {
                            theta_diag = std::max(theta_diag, std::abs(v));
                        } else {
                            theta_off += v * v;
                        }
                    }
                }
                theta_fro = std::sqrt(theta_fro);
                theta_off = std::sqrt(theta_off);
                theta2 =
                    cb_gmres::symmetric_spectral_norm(std::move(sm), gram_n);
            }
            cycle_records.push_back(cb_gmres::cycle_record{
                solve_index,
                cycle_index++,
                cycle_start_iter,
                end_iter - cycle_start_iter,
                reason,
                res_start,
                res_end,
                reorth_passes - cycle_start_reorth,
                psi_stats.max,
                psi_stats.count > 0 ? psi_stats.sum / psi_stats.count : 0.0,
                static_cast<int>(psi_stats.count),
                psi_stats.lb_max,
                psi_stats.count > 0 ? psi_stats.lb_sum / psi_stats.count : 0.0,
                psi_stats.chi_max,
                psi_stats.count > 0 ? psi_stats.chi_sum / psi_stats.count : 0.0,
                vinf_max,
                vinf_count > 0 ? vinf_sum / vinf_count : 0.0,
                vinf_count,
                theta2,
                theta_fro,
                theta_diag,
                theta_off,
                gram_n,
                psi_stats.loc_max,
                psi_stats.loc_count > 0
                    ? psi_stats.loc_sum / psi_stats.loc_count
                    : 0.0,
                static_cast<int>(psi_stats.loc_count)});
            cycle_start_iter = end_iter;
            cycle_start_reorth = reorth_passes;
            // Per cycle, unlike reorth_passes, which accumulates over the
            // solve.
            psi_stats = kernels::cb_gmres_psi_stats{};
            gram_n = 0;
            std::fill(gram.begin(), gram.end(), 0.0);
        };
        // True once every not-yet-converged right-hand side has met the
        // tolerance within the current cycle.
        auto cycle_tolerance_met = [&](const VectorNorms* res_norm) {
            if (!use_restart_ratio) {
                return false;
            }
            host_residual_norm->copy_from(res_norm);
            for (size_type i = 0; i < num_rhs; ++i) {
                if (fully_converged_rhs.get_const_data()[i]) {
                    continue;
                }
                if (host_residual_norm->at(0, i) >
                    restart_ratio * cycle_start_norm->at(0, i)) {
                    return false;
                }
            }
            return true;
        };
        // The first cycle starts from the residual computed above.
        record_cycle_start(residual_norm.get());
        // Start only after this value with performing forced iterations after
        // convergence detection
        constexpr int start_force_reset{10};
        bool perform_reset{false};
        // Fraction of the krylov_dim (or total_iter if it is lower),
        // determining the number of forced iteration to perform
        constexpr size_type forced_iteration_fraction{10};
        const size_type forced_limit{krylov_dim / forced_iteration_fraction};
        // Counter for the forced iterations. Start at max in order to properly
        // test convergence at the beginning
        size_type forced_iterations{forced_limit};

        while (true) {
            ++total_iter;
            // In the beginning, only force a fraction of the total iterations
            if (forced_iterations < forced_limit &&
                forced_iterations < total_iter / forced_iteration_fraction) {
                this->template log<log::Logger::iteration_complete>(
                    this, dense_b, dense_x, total_iter, residual.get(),
                    residual_norm.get(), nullptr, &stop_status, false);
                ++forced_iterations;
            } else {
                bool all_changed = stop_criterion->update()
                                       .num_iterations(total_iter)
                                       .residual(residual)
                                       .residual_norm(residual_norm)
                                       .solution(dense_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                this->template log<log::Logger::iteration_complete>(
                    this, dense_b, dense_x, total_iter, residual.get(),
                    residual_norm.get(), nullptr, &stop_status, all_changed);
                if (one_changed || all_changed) {
                    host_stop_status = stop_status;
                    bool host_array_changed{false};
                    for (size_type i = 0; i < host_stop_status.get_size();
                         ++i) {
                        auto local_status = host_stop_status.get_data() + i;
                        // Ignore all actually converged ones!
                        if (fully_converged_rhs.get_data()[i]) {
                            continue;
                        }
                        if (local_status->has_converged()) {
                            // If convergence was detected earlier, or
                            // at the very beginning:
                            if (stop_encountered_rhs.get_data()[i] ||
                                total_iter < start_force_reset) {
                                fully_converged_rhs.get_data()[i] = true;
                            } else {
                                stop_encountered_rhs.get_data()[i] = true;
                                local_status->reset();
                                host_array_changed = true;
                            }
                        }
                    }
                    if (host_array_changed) {
                        perform_reset = true;
                        pending_reason = "stop_reset";
                        stop_status = host_stop_status;
                    } else {
                        // Stop here can happen if all RHS are "fully_converged"
                        // or if it was stopped for non-convergence reason
                        // (like time or iteration)
                        break;
                    }
                    forced_iterations = 0;

                } else {
                    for (size_type i = 0; i < stop_encountered_rhs.get_size();
                         ++i) {
                        stop_encountered_rhs.get_data()[i] = false;
                    }
                }
            }

            if (perform_reset || restart_iter == krylov_dim) {
                // Log the cycle that is about to be closed before any state is
                // reset. When perform_reset is false the cap was reached.
                close_cycle(perform_reset ? pending_reason : "cap", total_iter);
                pending_reason = "cap";
                perform_reset = false;
                // Restart
                // use a view in case this is called earlier
                auto hessenberg_view = hessenberg->create_submatrix(
                    span{0, restart_iter}, span{0, num_rhs * (restart_iter)});

                exec->run(cb_gmres::make_solve_krylov(
                    residual_norm_collection->get_const_device_view(),
                    krylov_bases_range.get_accessor().to_const(),
                    hessenberg_view->get_const_device_view(),
                    y->get_device_view(),
                    before_preconditioner->get_device_view(), final_iter_nums));
                // Solve upper triangular.
                // y = hessenberg \ residual_norm_collection

                this->get_preconditioner()->apply(before_preconditioner,
                                                  after_preconditioner);
                dense_x->add_scaled(one_op, after_preconditioner);
                // Solve x
                // x = x + get_preconditioner() * krylov_bases * y
                residual->copy_from(dense_b);
                // residual = dense_b
                this->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                                 residual);
                // residual = residual - Ax
                exec->run(cb_gmres::make_restart(
                    residual->get_const_device_view(),
                    residual_norm->get_device_view(),
                    residual_norm_collection->get_device_view(),
                    arnoldi_norm->get_device_view(), krylov_bases_range,
                    next_krylov_basis->get_device_view(), final_iter_nums,
                    reduction_tmp, krylov_dim));
                // residual_norm = norm(residual)
                // residual_norm_collection = {residual_norm, 0, ..., 0}
                // krylov_bases(:, 1) = residual / residual_norm
                // next_krylov_basis = residual / residual_norm
                // final_iter_nums = {0, ..., 0}
                restart_iter = 0;
                record_cycle_start(residual_norm.get());
            }

            this->get_preconditioner()->apply(next_krylov_basis,
                                              preconditioned_vector);
            // preconditioned_vector = get_preconditioner() *
            // next_krylov_basis

            // Do Arnoldi and givens rotation
            auto hessenberg_iter = hessenberg->create_submatrix(
                span{0, restart_iter + 2},
                span{num_rhs * restart_iter, num_rhs * (restart_iter + 1)});
            auto buffer_iter = buffer->create_submatrix(
                span{0, restart_iter + 2}, span{0, num_rhs});

            // Start of arnoldi
            this->get_system_matrix()->apply(preconditioned_vector,
                                             next_krylov_basis);
            // next_krylov_basis = A * preconditioned_vector
            exec->run(cb_gmres::make_arnoldi(
                next_krylov_basis->get_device_view(),
                givens_sin->get_device_view(), givens_cos->get_device_view(),
                residual_norm->get_device_view(),
                residual_norm_collection->get_device_view(), krylov_bases_range,
                hessenberg_iter->get_device_view(),
                buffer_iter->get_device_view(), arnoldi_norm->get_device_view(),
                restart_iter, final_iter_nums, stop_status, reorth_status,
                num_reorth, this->get_max_reorth_steps(), reorth_counter,
                psi_counter));
            // The Arnoldi step has stored vector restart_iter + 1, so its
            // column of the Gram matrix can be formed now.
            record_gram_column(restart_iter + 1);
            // for i in 0:restart_iter
            //     hessenberg(restart_iter, i) = next_krylov_basis' *
            //     krylov_bases(:, i) next_krylov_basis  -=
            //     hessenberg(restart_iter, i) * krylov_bases(:, i)
            // end
            // hessenberg(restart_iter, restart_iter + 1) =
            // norm(next_krylov_basis) next_krylov_basis /=
            // hessenberg(restart_iter, restart_iter + 1) End of arnoldi
            // Start apply givens rotation for j in 0:restart_iter
            //     temp             =  cos(j)*hessenberg(j) +
            //                         sin(j)*hessenberg(j+1)
            //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
            //                         cos(j)*hessenberg(j+1)
            //     hessenberg(j)    =  temp;
            // end
            // Calculate sin and cos
            // hessenberg(restart_iter)   =
            // cos(restart_iter)*hessenberg(restart_iter) +
            //                      sin(restart_iter)*hessenberg(restart_iter)
            // hessenberg(restart_iter+1) = 0
            // End apply givens rotation
            // Calculate residual norm

            restart_iter++;

            // residual_norm now holds the residual of the correction system.
            // Testing it here takes the restart at the top of the next
            // iteration, where the solution update and the explicit residual
            // recomputation already happen.
            if (restart_iter < krylov_dim &&
                cycle_tolerance_met(residual_norm.get())) {
                perform_reset = true;
                pending_reason = "tolerance";
            }
        }  // closes while(true)

        // The final cycle ends by leaving the loop rather than by restarting.
        close_cycle("solve_end", total_iter);
        // Solve x

        auto hessenberg_small = hessenberg->create_submatrix(
            span{0, restart_iter}, span{0, num_rhs * restart_iter});

        exec->run(cb_gmres::make_solve_krylov(
            residual_norm_collection->get_const_device_view(),
            krylov_bases_range.get_accessor().to_const(),
            hessenberg_small->get_const_device_view(), y->get_device_view(),
            before_preconditioner->get_device_view(), final_iter_nums));
        // Solve upper triangular.
        // y = hessenberg \ residual_norm_collection
        this->get_preconditioner()->apply(before_preconditioner,
                                          after_preconditioner);
        dense_x->add_scaled(one_op, after_preconditioner);
        // Solve x
        // x = x + get_preconditioner() * krylov_bases * y

        cb_gmres::write_cycle_log(cycle_records);
    };  // End of apply_lambda

    // Look which precision to use as the storage type
    helper<ValueType>::call(apply_templated, this->get_storage_precision());
}


template <typename ValueType>
void CbGmres<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                    const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}

#define GKO_DECLARE_CB_GMRES(_type1) class CbGmres<_type1>
#define GKO_DECLARE_CB_GMRES_TRAITS(_type1) \
    struct workspace_traits<CbGmres<_type1>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_BASE(GKO_DECLARE_CB_GMRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_BASE(GKO_DECLARE_CB_GMRES_TRAITS);


}  // namespace solver
}  // namespace gko
