// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <map>
#include <span>
#include <spblas/spblas.hpp>
#include <string>


#include <omp.h>


#include <ginkgo/ginkgo.hpp>


// Creates a stencil matrix in CSR format for the given number of discretization
// points. Assume they have proper memory allocation
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(unsigned int discretization_points,
                             IndexType* row_ptrs, IndexType* col_idxs,
                             ValueType* values)
{
    IndexType pos = 0;
    const ValueType coefs[] = {-1, 2, -1};
    row_ptrs[0] = pos;
    for (int i = 0; i < discretization_points; ++i) {
        for (auto ofs : {-1, 0, 1}) {
            if (0 <= i + ofs && i + ofs < discretization_points) {
                values[pos] = coefs[ofs + 1];
                col_idxs[pos] = i + ofs;
                ++pos;
            }
        }
        row_ptrs[i + 1] = pos;
    }
}


template <typename ValueType, typename IndexType>
class SpblasCsr
    : public gko::EnableLinOp<SpblasCsr<ValueType, IndexType>>,
      public gko::EnableCreateMethod<SpblasCsr<ValueType, IndexType>> {
public:
    // This constructor will be called by the create method. Here we initialize
    // the coefficients of the stencil.
    SpblasCsr(std::shared_ptr<const gko::Executor> exec, unsigned int size = 0)
        : gko::EnableLinOp<SpblasCsr>(exec, gko::dim<2>{size}),
          nnz((size != 0) * (3 * size - 2)),
          values_(exec, nnz),
          col_idxs_(exec, nnz),
          row_ptrs_(exec, size + 1)
    {
        if (size > 0) {
            generate_stencil_matrix(size, row_ptrs_.get_data(),
                                    col_idxs_.get_data(), values_.get_data());
        }
    }

protected:
    using vec = gko::matrix::Dense<ValueType>;

    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        // we only implement the operator for dense RHS on reference
        // gko::as will throw an exception if its argument is not Dense.
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        // pointer need non-const?
        spblas::csr_view<ValueType, IndexType> a(
            const_cast<ValueType*>(values_.get_const_data()),
            const_cast<IndexType*>(row_ptrs_.get_const_data()),
            const_cast<IndexType*>(col_idxs_.get_const_data()),
            spblas::index<IndexType>(this->get_size()[0], this->get_size()[1]),
            static_cast<IndexType>(nnz));
        std::span<ValueType> b_span(
            const_cast<ValueType*>(dense_b->get_const_values()),
            dense_b->get_num_stored_elements());
        std::span<ValueType> x_span(dense_x->get_values(),
                                    dense_x->get_num_stored_elements());
        dense_x->fill(0.0);
        spblas::multiply(a, b_span, x_span);
    }

    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto tmp_x = dense_x->clone();
        this->apply_impl(b, tmp_x.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, tmp_x);
    }

private:
    gko::size_type nnz;
    gko::array<ValueType> values_;
    gko::array<IndexType> col_idxs_;
    gko::array<IndexType> row_ptrs_;
};


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ValueType>
void generate_rhs(Closure f, ValueType u0, ValueType u1,
                  gko::matrix::Dense<ValueType>* rhs)
{
    const auto discretization_points = rhs->get_size()[0];
    auto values = rhs->get_values();
    const ValueType h = 1.0 / (discretization_points + 1);
    for (int i = 0; i < discretization_points; ++i) {
        const ValueType xi = ValueType(i + 1) * h;
        values[i] = -f(xi) * h * h;
    }
    values[0] += u0;
    values[discretization_points - 1] += u1;
}


// Prints the solution `u`.
template <typename ValueType>
void print_solution(ValueType u0, ValueType u1,
                    const gko::matrix::Dense<ValueType>* u)
{
    std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    std::cout << u1 << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType>
double calculate_error(int discretization_points,
                       const gko::matrix::Dense<ValueType>* u,
                       Closure correct_u)
{
    const auto h = 1.0 / (discretization_points + 1);
    auto error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        const auto xi = (i + 1) * h;
        error +=
            abs(u->get_const_values()[i] - correct_u(xi)) / abs(correct_u(xi));
    }
    return error;
}


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [discretization_points = 100]"
                  << std::endl;
        std::exit(-1);
    }

    const unsigned int discretization_points =
        argc >= 2 ? std::atoi(argv[1]) : 100u;
    // executor used by the application
    const auto app_exec = gko::ReferenceExecutor::create();

    // problem:
    auto correct_u = [](ValueType x) { return x * x * x; };
    auto f = [](ValueType x) { return ValueType{6} * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // initialize vectors
    auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    generate_rhs(f, u0, u1, rhs.get());
    auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    for (int i = 0; i < u->get_size()[0]; ++i) {
        u->get_values()[i] = 0.0;
    }

    const RealValueType reduction_factor{1e-7};
    // Generate solver and solve the system
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(discretization_points),
            gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(
                reduction_factor))
        .on(app_exec)
        // notice how our custom matrix can be used in the same way as
        // any built-in type
        ->generate(SpblasCsr<ValueType, IndexType>::create(
            app_exec, discretization_points))
        ->apply(rhs, u);
    std::cout << "\nSolve complete."
              << "\nThe average relative error is "
              << calculate_error(discretization_points, u.get(), correct_u) /
                     discretization_points
              << std::endl;
}
