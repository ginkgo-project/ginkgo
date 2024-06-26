// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/cb_gmres.hpp>


#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <libpressio_ext/cpp/json.h>
#include <libpressio_ext/cpp/libpressio.h>
#include <libpressio_meta.h>  //provides frsz
#include <string>
#include <type_traits>
#include <vector>


#include <nlohmann/json.hpp>


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
#include "core/solver/cb_gmres_accessor.hpp"
#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace solver {
namespace cb_gmres {
namespace {


GKO_REGISTER_OPERATION(initialize, cb_gmres::initialize);
GKO_REGISTER_OPERATION(restart, cb_gmres::restart);
GKO_REGISTER_OPERATION(arnoldi, cb_gmres::arnoldi);
GKO_REGISTER_OPERATION(solve_krylov, cb_gmres::solve_krylov);

GKO_REGISTER_OPERATION(restart_f, cb_gmres::restart_f);
GKO_REGISTER_OPERATION(arnoldi_f, cb_gmres::arnoldi_f);
GKO_REGISTER_OPERATION(solve_krylov_f, cb_gmres::solve_krylov_f);


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
        case cb_gmres::storage_precision::use_pressio:
            callable(ValueType{1});
            break;
        case cb_gmres::storage_precision::use_frsz2_21:
            callable(ValueType{21});
            break;
        case cb_gmres::storage_precision::use_frsz2_32:
            callable(ValueType{32});
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
        case cb_gmres::storage_precision::use_pressio:
        case cb_gmres::storage_precision::use_frsz2_21:
        case cb_gmres::storage_precision::use_frsz2_32:
            GKO_NOT_SUPPORTED(st);
            break;
        default:
            callable(ValueType{});
        }
    }
};


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


bool check_for_pressio(double value) { return value == 1.0; }
bool check_for_pressio(float value) { return value == 1.0; }

template <typename T>
bool check_for_pressio(T value)
{
    return false;
}


int check_for_frsz2(double value)
{
    return value == 21.0 ? 21 : value == 32.0 ? 32 : 0;
}
int check_for_frsz2(float value)
{
    return value == 21.0 ? 21 : value == 32.0 ? 32 : 0;
}

template <typename T>
int check_for_frsz2(T value)
{
    return 0;
}


// ADDED
template <typename RangeHelper>
void compress_data(RangeHelper&& helper, std::vector<pressio_data>& p_data_vec,
                   pressio_data& temp)
{}

template <typename ValueType, typename StorageType>
struct compression_helper {
    compression_helper(bool use_compr, size_type num_rows, size_type num_vecs,
                       std::function<void(void*)> init_compressor)
        : use_compr_{use_compr},
          num_rows_{num_rows},
          uncompressed_size_{},
          compressed_size_{},
          pc_{},
          in_temp_{},
          out_temp_{},
          p_data_vec_(use_compr_ ? 1 : 0)
    {
        using namespace std::string_literals;
        if (use_compr_) {
            libpressio_register_all();
            init_compressor(&pc_);
            pc_->set_options({{"pressio:metric", "size"}});
            const auto pressio_type = std::is_same<ValueType, float>::value
                                          ? pressio_float_dtype
                                          : pressio_double_dtype;
            for (size_type i = 0; i < p_data_vec_.size(); ++i) {
                p_data_vec_[i] = pressio_data::empty(pressio_byte_dtype, {});
            }
            in_temp_ = pressio_data::owning(pressio_type, {num_rows});
            out_temp_ = pressio_data::owning(pressio_type, {num_rows});
        }
    }

    void compress(size_type krylov_idx,
                  gko::cb_gmres::Range3dHelper<ValueType, StorageType>& rhelper)
    {
        if (!use_compr_) {
            return;
        }
        GKO_ASSERT(rhelper.get_range().length(2) == 1);

        const auto exec = rhelper.get_bases().get_executor().get();
        const auto host_exec = exec->get_master().get();

        // Reinterpret_cast necessary for type check if no compressor is used
        auto raw_krylov_base = reinterpret_cast<ValueType*>(
            rhelper.get_bases().get_data() + krylov_idx * num_rows_);
        host_exec->copy_from(exec, num_rows_, raw_krylov_base,
                             reinterpret_cast<ValueType*>(in_temp_.data()));
        if (pc_->compress(&in_temp_, &p_data_vec_[0])) {
            std::cerr << pc_->error_msg() << '\n';
        }
        if (pc_->decompress(&p_data_vec_[0], &out_temp_)) {
            std::cerr << pc_->error_msg() << '\n';
        }
        uncompressed_size_ += pc_->get_metrics_results()
                                  .get("size:uncompressed_size")
                                  .template get<std::uint64_t>()
                                  .value_or(0);
        compressed_size_ += pc_->get_metrics_results()
                                .get("size:compressed_size")
                                .template get<std::uint64_t>()
                                .value_or(0);
        exec->copy_from(host_exec, num_rows_,
                        reinterpret_cast<const ValueType*>(out_temp_.data()),
                        raw_krylov_base);
    }

    void print_metrics() const
    {
        if (false && use_compr_) {
            std::cout << pc_->get_metrics_results() << '\n';
        }
    }

    void set_use_compressor(bool use_compr) { use_compr_ = false; }

    double get_average_bit_rate() const
    {
        return use_compr_ ? (CHAR_BIT * static_cast<double>(compressed_size_) /
                             static_cast<double>(uncompressed_size_ /
                                                 sizeof(ValueType)))
                          : static_cast<double>(CHAR_BIT * sizeof(StorageType));
    }


private:
    std::string compressor_;
    bool use_compr_;
    size_type num_rows_;
    size_type compressed_size_;
    size_type uncompressed_size_;
    pressio_compressor pc_;
    pressio_data in_temp_;
    pressio_data out_temp_;
    std::vector<pressio_data> p_data_vec_;
};

template <typename ValueType>
struct run_frsz2 {
public:
    run_frsz2(std::shared_ptr<const Executor> exec) : exec_(std::move(exec)) {}

    template <typename... Args>
    void restart_f(Args&&... args) const
    {
        exec_->run(cb_gmres::make_restart_f(std::forward<Args>(args)...));
    }

    template <typename... Args>
    void solve_krylov_f(Args&&... args) const
    {
        exec_->run(cb_gmres::make_solve_krylov_f(std::forward<Args>(args)...));
    }

    template <typename... Args>
    void arnoldi_f(Args&&... args) const
    {
        exec_->run(cb_gmres::make_arnoldi_f(std::forward<Args>(args)...));
    }

private:
    std::shared_ptr<const Executor> exec_;
};


template <typename NcValueType>
struct run_frsz2<std::complex<NcValueType>> {
public:
    run_frsz2(std::shared_ptr<const Executor>) {}

    template <typename... Args>
    void restart_f(Args&&... args) const
    {
        GKO_NOT_IMPLEMENTED;
    }

    template <typename... Args>
    void solve_krylov_f(Args&&... args) const
    {
        GKO_NOT_IMPLEMENTED;
    }

    template <typename... Args>
    void arnoldi_f(Args&&... args) const
    {
        GKO_NOT_IMPLEMENTED;
    }
};


template <typename Range>
void print_krylov_vectors(Range&& curr, size_type num_vecs, size_type iteration)
{
    using T = typename std::decay_t<Range>::accessor::arithmetic_type;
    const std::string matrix_delim = "$$$$$$$$$$\n";
    const std::string iteration_prefix = "__iteration: ";
    const std::string elem_delim = " ";
    std::cout << std::setprecision(17) << std::scientific;
    std::cout << iteration_prefix << iteration << '\n';
    std::cout << matrix_delim;
    for (size_type i = 0; i < num_vecs; ++i) {
        for (size_type row = 0; row < curr.length(1); ++row) {
            const T cv = curr(i, row, 0);
            std::cout << cv << elem_delim;
        }
        std::cout << '\n';
    }
    std::cout << matrix_delim;
}


template <typename ValueType, typename StorageType>
struct krylov_basis_helper {
    static std::unique_ptr<matrix::Dense<ValueType>> extract(
        gko::cb_gmres::Range3dHelper<ValueType, StorageType>&, size_type, bool)
    {
        return nullptr;
    }
};


template <typename ValueType>
struct krylov_basis_helper<ValueType, ValueType> {
    static std::unique_ptr<matrix::Dense<ValueType>> extract(
        gko::cb_gmres::Range3dHelper<ValueType, ValueType>& helper,
        size_type local_iter, bool do_store)
    {
        std::unique_ptr<matrix::Dense<ValueType>> krylov_basis{nullptr};
        if (do_store && helper.get_range().length(2) == 1) {
            const auto& range = helper.get_range();
            const auto source_exec = helper.get_bases().get_executor();
            const auto dest_exec = source_exec->get_master();
            const auto number_vectors =
                std::min<size_type>(local_iter + 1, range.length(0));
            krylov_basis = matrix::Dense<ValueType>::create(
                dest_exec, dim<2>{number_vectors, range.length(1)});
            dest_exec->copy_from(source_exec, number_vectors * range.length(1),
                                 helper.get_bases().get_const_data(),
                                 krylov_basis->get_values());
        }
        return krylov_basis;
    }
};


template <typename ValueType>
void CbGmres<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    // Current workaround to get a lambda with a template argument (only
    // the type of `value` matters, the content does not)
    auto apply_templated = [&](auto value) {
        using storage_type = decltype(value);
        using rc_value_type = remove_complex<ValueType>;
        const bool use_pressio = check_for_pressio(value);
        const auto which_frsz2 = check_for_frsz2(value);

        using Vector = matrix::Dense<ValueType>;
        using VectorNorms = matrix::Dense<rc_value_type>;
        using Range3dHelper =
            gko::cb_gmres::Range3dHelper<ValueType, storage_type>;
        using Frsz2Compressor21 = acc::frsz2<21, 32, rc_value_type>;
        using Frsz2Compressor32 = acc::frsz2<32, 32, rc_value_type>;
        array<uint8> compressed_storage(this->get_executor());


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

        // Added
        const std::array<acc::size_type, 3> krylov_bases_dim_a{
            static_cast<acc::size_type>(krylov_bases_dim[0]),
            static_cast<acc::size_type>(krylov_bases_dim[1]),
            static_cast<acc::size_type>(krylov_bases_dim[2])};
        if (which_frsz2 == 21) {
            helper.get_bases().clear();
            compressed_storage.resize_and_reset(
                Frsz2Compressor21::memory_requirement(krylov_bases_dim_a));
            this->average_bit_rate_ =
                static_cast<double>(compressed_storage.get_size() * CHAR_BIT) /
                (krylov_bases_dim[0] * krylov_bases_dim[1] *
                 krylov_bases_dim[2]);
        } else if (which_frsz2 == 32) {
            helper.get_bases().clear();
            compressed_storage.resize_and_reset(
                Frsz2Compressor32::memory_requirement(krylov_bases_dim_a));
            this->average_bit_rate_ =
                static_cast<double>(compressed_storage.get_size() * CHAR_BIT) /
                (krylov_bases_dim[0] * krylov_bases_dim[1] *
                 krylov_bases_dim[2]);
        } else if (which_frsz2 == 0) {
            // Valid state which will be processed later on
        } else {
            GKO_NOT_IMPLEMENTED;
        }
        Frsz2Compressor21 krylov_bases_frsz2_21(krylov_bases_dim_a,
                                                compressed_storage.get_data());
        Frsz2Compressor32 krylov_bases_frsz2_32(krylov_bases_dim_a,
                                                compressed_storage.get_data());
        run_frsz2<ValueType> run_f_helper(this->get_executor());

        // ADDED
        compression_helper<ValueType, storage_type> comp_helper(
            use_pressio, num_rows, krylov_dim + 1, parameters_.init_compressor);

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
        // ADDED stop_compression
        // auto residual_norm_host =
        //     VectorNorms::create(exec->get_master(), dim<2>{1, num_rhs});
        // auto last_residual_norm_host =
        //     VectorNorms::create(exec->get_master(), dim<2>{1, num_rhs});
        // 1st row of arnoldi_norm: == eta * norm2(old_next_krylov_basis)
        //                          with eta == 1 / sqrt(2)
        //                          (computed right before updating
        //                          next_krylov_basis)
        // 2nd row of arnoldi_norm: The actual arnoldi norm
        //                          == norm2(next_krylov_basis)
        // 3rd row of arnoldi_norm: the infinity norm of next_krylov_basis
        //                          (ONLY when using a scalar accessor)
        auto arnoldi_norm = VectorNorms::create(exec, dim<2>{3, num_rhs});
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
        exec->run(cb_gmres::make_initialize(dense_b, residual.get(),
                                            givens_sin.get(), givens_cos.get(),
                                            &stop_status, krylov_dim));
        // residual = dense_b
        // givens_sin = givens_cos = 0
        this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
        // residual = residual - Ax

        if (which_frsz2 == 0) {
            exec->run(cb_gmres::make_restart(
                residual.get(), residual_norm.get(),
                residual_norm_collection.get(), arnoldi_norm.get(),
                krylov_bases_range, next_krylov_basis.get(), &final_iter_nums,
                reduction_tmp, krylov_dim));
            comp_helper.compress(0, helper);  // ADDED
        } else if (which_frsz2 == 21) {
            run_f_helper.restart_f(residual.get(), residual_norm.get(),
                                   residual_norm_collection.get(),
                                   arnoldi_norm.get(), krylov_bases_frsz2_21,
                                   next_krylov_basis.get(), &final_iter_nums,
                                   reduction_tmp, krylov_dim);
        } else if (which_frsz2 == 32) {
            run_f_helper.restart_f(residual.get(), residual_norm.get(),
                                   residual_norm_collection.get(),
                                   arnoldi_norm.get(), krylov_bases_frsz2_32,
                                   next_krylov_basis.get(), &final_iter_nums,
                                   reduction_tmp, krylov_dim);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
        // ADDED stop_compression
        // residual_norm_host->copy_from(residual_norm.get());
        // last_residual_norm_host->copy_from(residual_norm_host.get());
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
            // ADDED stop_compression
            // if (total_iter > 0) {
            //     for (size_type i = 0; i < num_rhs; ++i) {
            //         if (last_residual_norm_host->at(0, i) <=
            //             residual_norm_host->at(0, i)) {
            //             comp_helper.set_use_compressor(false);
            //         }
            //     }
            // }

            if (perform_reset || restart_iter == krylov_dim) {
                perform_reset = false;
                // Restart
                // use a view in case this is called earlier
                auto hessenberg_view = hessenberg->create_submatrix(
                    span{0, restart_iter}, span{0, num_rhs * (restart_iter)});

                // ADDED stop_compression
                // last_residual_norm_host->copy_from(residual_norm_host.get());

                // print_krylov_vectors(krylov_bases_range, restart_iter + 1,
                //                     total_iter);
                auto krylov_basis_as_mtx =
                    krylov_basis_helper<ValueType, storage_type>::extract(
                        helper, restart_iter, which_frsz2 == 0 && !use_pressio);
                // Only write log if both the log and the matrix actually exists
                if (this->parameters_.krylov_basis_log && krylov_basis_as_mtx) {
                    this->parameters_.krylov_basis_log->operator[](total_iter) =
                        std::move(krylov_basis_as_mtx);
                }
                if (which_frsz2 == 0) {
                    exec->run(cb_gmres::make_solve_krylov(
                        residual_norm_collection.get(),
                        krylov_bases_range.get_accessor().to_const(),
                        hessenberg_view.get(), y.get(),
                        before_preconditioner.get(), &final_iter_nums));
                } else if (which_frsz2 == 21) {
                    run_f_helper.solve_krylov_f(
                        residual_norm_collection.get(), krylov_bases_frsz2_21,
                        hessenberg_view.get(), y.get(),
                        before_preconditioner.get(), &final_iter_nums);
                } else if (which_frsz2 == 32) {
                    run_f_helper.solve_krylov_f(
                        residual_norm_collection.get(), krylov_bases_frsz2_32,
                        hessenberg_view.get(), y.get(),
                        before_preconditioner.get(), &final_iter_nums);
                } else {
                    GKO_NOT_IMPLEMENTED;
                }
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
                if (which_frsz2 == 0) {
                    exec->run(cb_gmres::make_restart(
                        residual.get(), residual_norm.get(),
                        residual_norm_collection.get(), arnoldi_norm.get(),
                        krylov_bases_range, next_krylov_basis.get(),
                        &final_iter_nums, reduction_tmp, krylov_dim));
                } else if (which_frsz2 == 21) {
                    run_f_helper.restart_f(
                        residual.get(), residual_norm.get(),
                        residual_norm_collection.get(), arnoldi_norm.get(),
                        krylov_bases_frsz2_21, next_krylov_basis.get(),
                        &final_iter_nums, reduction_tmp, krylov_dim);
                } else if (which_frsz2 == 32) {
                    run_f_helper.restart_f(
                        residual.get(), residual_norm.get(),
                        residual_norm_collection.get(), arnoldi_norm.get(),
                        krylov_bases_frsz2_32, next_krylov_basis.get(),
                        &final_iter_nums, reduction_tmp, krylov_dim);
                } else {
                    GKO_NOT_IMPLEMENTED;
                }
                // residual_norm = norm(residual)
                // residual_norm_collection = {residual_norm, 0, ..., 0}
                // krylov_bases(:, 1) = residual / residual_norm
                // next_krylov_basis = residual / residual_norm
                // final_iter_nums = {0, ..., 0}
                // ADDED stop_compression
                // residual_norm_host->copy_from(residual_norm.get());
                comp_helper.compress(0, helper);  // ADDED
                restart_iter = 0;
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
            if (which_frsz2 == 0) {
                exec->run(cb_gmres::make_arnoldi(
                    next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                    residual_norm.get(), residual_norm_collection.get(),
                    krylov_bases_range, hessenberg_iter.get(),
                    buffer_iter.get(), arnoldi_norm.get(), restart_iter,
                    &final_iter_nums, &stop_status, &reorth_status, &num_reorth,
                    this->parameters_.detail_operation_logger.get()));
            } else if (which_frsz2 == 21) {
                run_f_helper.arnoldi_f(
                    next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                    residual_norm.get(), residual_norm_collection.get(),
                    krylov_bases_frsz2_21, hessenberg_iter.get(),
                    buffer_iter.get(), arnoldi_norm.get(), restart_iter,
                    &final_iter_nums, &stop_status, &reorth_status, &num_reorth,
                    this->parameters_.detail_operation_logger.get());
            } else if (which_frsz2 == 32) {
                run_f_helper.arnoldi_f(
                    next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                    residual_norm.get(), residual_norm_collection.get(),
                    krylov_bases_frsz2_32, hessenberg_iter.get(),
                    buffer_iter.get(), arnoldi_norm.get(), restart_iter,
                    &final_iter_nums, &stop_status, &reorth_status, &num_reorth,
                    this->parameters_.detail_operation_logger.get());
            } else {
                GKO_NOT_IMPLEMENTED;
            }
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

            comp_helper.compress(restart_iter + 1, helper);  // ADDED
            restart_iter++;
        }  // closes while(true)
        // Solve x

        auto hessenberg_small = hessenberg->create_submatrix(
            span{0, restart_iter}, span{0, num_rhs * restart_iter});

        auto krylov_basis_as_mtx =
            krylov_basis_helper<ValueType, storage_type>::extract(
                helper, restart_iter, which_frsz2 == 0 && !use_pressio);
        // Only write log if both the log and the matrix actually exists
        if (this->parameters_.krylov_basis_log && krylov_basis_as_mtx) {
            this->parameters_.krylov_basis_log->operator[](total_iter) =
                std::move(krylov_basis_as_mtx);
        }
        if (which_frsz2 == 0) {
            exec->run(cb_gmres::make_solve_krylov(
                residual_norm_collection.get(),
                krylov_bases_range.get_accessor().to_const(),
                hessenberg_small.get(), y.get(), before_preconditioner.get(),
                &final_iter_nums));
        } else if (which_frsz2 == 21) {
            run_f_helper.solve_krylov_f(
                residual_norm_collection.get(), krylov_bases_frsz2_21,
                hessenberg_small.get(), y.get(), before_preconditioner.get(),
                &final_iter_nums);
        } else if (which_frsz2 == 32) {
            run_f_helper.solve_krylov_f(
                residual_norm_collection.get(), krylov_bases_frsz2_32,
                hessenberg_small.get(), y.get(), before_preconditioner.get(),
                &final_iter_nums);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
        // Solve upper triangular.
        // y = hessenberg \ residual_norm_collection
        this->get_preconditioner()->apply(before_preconditioner,
                                          after_preconditioner);
        dense_x->add_scaled(one_op, after_preconditioner);
        // Solve x
        // x = x + get_preconditioner() * krylov_bases * y
        comp_helper.print_metrics();  // ADDED
        if (which_frsz2 == 0) {
            this->average_bit_rate_ = comp_helper.get_average_bit_rate();
        }
        // If it was using frsz2, it was already set
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
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_TRAITS);


}  // namespace solver
}  // namespace gko
