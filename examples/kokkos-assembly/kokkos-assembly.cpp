// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include <ginkgo/ginkgo.hpp>

#include <ginkgo/extensions/kokkos.hpp>


namespace gko::ext::kokkos::detail {


/**
 * Specialization of type mapper for gko::device_matrix_data.
 *
 * @tparam ValueType  The value type of the matrix elements
 * @tparam IndexType   The index type of the matrix elements
 * @tparam MemorySpace  The Kokkos memory space to use.
 */
template <typename ValueType, typename IndexType, typename MemorySpace>
struct mapper<device_matrix_data<ValueType, IndexType>, MemorySpace> {
    using index_mapper = mapper<array<IndexType>, MemorySpace>;
    using value_mapper = mapper<array<ValueType>, MemorySpace>;

    /**
     * This struct defines the layout of the device_matrix_data type in terms
     * of arrays.
     *
     * @tparam ValueType_c  The value type of the matrix elements, might have
     *                      other cv qualifiers than ValueType
     * @tparam IndexType_c  The index type of the matrix elements, might have
     *                      other cv qualifiers than IndexType
     */
    template <typename ValueType_c, typename IndexType_c>
    struct type {
        using index_array = typename index_mapper::template type<IndexType_c>;
        using value_array = typename value_mapper::template type<ValueType_c>;

        /**
         * Constructor based on size and raw pointers
         *
         * @param size  The number of stored elements
         * @param row_idxs  Pointer to the row indices
         * @param col_idxs  Pointer to the column indices
         * @param values  Pointer to the values
         *
         * @return  An object which has each gko::array of the
         *          device_matrix_data mapped to a Kokkos view
         */
        static type map(size_type size, IndexType_c* row_idxs,
                        IndexType_c* col_idxs, ValueType_c* values)
        {
            return {index_mapper::map(row_idxs, size),
                    index_mapper::map(col_idxs, size),
                    value_mapper::map(values, size)};
        }

        index_array row_idxs;
        index_array col_idxs;
        value_array values;
    };

    static type<ValueType, IndexType> map(
        device_matrix_data<ValueType, IndexType>& md)
    {
        assert_compatibility<MemorySpace>(md);
        return type<ValueType, IndexType>::map(
            md.get_num_stored_elements(), md.get_row_idxs(), md.get_col_idxs(),
            md.get_values());
    }

    static type<const ValueType, const IndexType> map(
        const device_matrix_data<ValueType, IndexType>& md)
    {
        assert_compatibility<MemorySpace>(md);
        return type<const ValueType, const IndexType>::map(
            md.get_num_stored_elements(), md.get_const_row_idxs(),
            md.get_const_col_idxs(), md.get_const_values());
    }
};


}  // namespace gko::ext::kokkos::detail


// Creates a stencil matrix in CSR format for the given number of discretization
// points.
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(gko::matrix::Csr<ValueType, IndexType>* matrix)
{
    auto exec = matrix->get_executor();
    const auto discretization_points = matrix->get_size()[0];

    // Over-allocate storage for the matrix elements. Each row has 3 entries,
    // except for the first and last one. To handle each row uniformly, we
    // allocate space for 3x the number of rows.
    gko::device_matrix_data<ValueType, IndexType> md(exec, matrix->get_size(),
                                                     discretization_points * 3);

    // Create Kokkos views on Ginkgo data.
    auto k_md = gko::ext::kokkos::map_data(md);

    // Create the matrix entries. This also creates zero entries for the
    // first and second row to handle all rows uniformly.
    Kokkos::parallel_for(
        "generate_stencil_matrix", md.get_num_stored_elements(),
        KOKKOS_LAMBDA(int i) {
            const ValueType coefs[] = {-1, 2, -1};
            auto ofs = static_cast<IndexType>((i % 3) - 1);
            auto row = static_cast<IndexType>(i / 3);
            auto col = row + ofs;

            // To prevent branching, a mask is used to set the entry to
            // zero, if the column is out-of-bounds
            auto mask =
                static_cast<IndexType>(0 <= col && col < discretization_points);

            k_md.row_idxs[i] = mask * row;
            k_md.col_idxs[i] = mask * col;
            k_md.values[i] = mask * coefs[ofs + 1];
        });

    // Add up duplicate (zero) entries.
    md.sum_duplicates();

    // Build Csr matrix.
    matrix->read(std::move(md));
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ValueType>
void generate_rhs(Closure&& f, ValueType u0, ValueType u1,
                  gko::matrix::Dense<ValueType>* rhs)
{
    const auto discretization_points = rhs->get_size()[0];
    auto k_rhs = gko::ext::kokkos::map_data(rhs);
    Kokkos::parallel_for(
        "generate_rhs", discretization_points, KOKKOS_LAMBDA(int i) {
            const ValueType h = 1.0 / (discretization_points + 1);
            const ValueType xi = ValueType(i + 1) * h;
            k_rhs(i, 0) = -f(xi) * h * h;
            if (i == 0) {
                k_rhs(i, 0) += u0;
            }
            if (i == discretization_points - 1) {
                k_rhs(i, 0) += u1;
            }
        });
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType>
double calculate_error(int discretization_points,
                       const gko::matrix::Dense<ValueType>* u,
                       Closure&& correct_u)
{
    auto k_u = gko::ext::kokkos::map_data(u);
    auto error = 0.0;
    Kokkos::parallel_reduce(
        "calculate_error", discretization_points,
        KOKKOS_LAMBDA(int i, double& lsum) {
            const auto h = 1.0 / (discretization_points + 1);
            const auto xi = (i + 1) * h;
            lsum += Kokkos::abs((k_u(i, 0) - correct_u(xi)) /
                                Kokkos::abs(correct_u(xi)));
        },
        error);
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
    using bj = gko::preconditioner::Jacobi<ValueType>;

    // Print help message. For details on the kokkos-options see
    // https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html#initialization-by-command-line-arguments
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [discretization_points] [kokkos-options]" << std::endl;
        Kokkos::ScopeGuard kokkos(argc, argv);  // print Kokkos help
        std::exit(1);
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    const auto discretization_points =
        static_cast<gko::size_type>(argc >= 2 ? std::atoi(argv[1]) : 100u);

    // chooses the executor that corresponds to the Kokkos DefaultExecutionSpace
    auto exec = gko::ext::kokkos::create_default_executor();

    // problem:
    auto correct_u = [] KOKKOS_FUNCTION(ValueType x) { return x * x * x; };
    auto f = [] KOKKOS_FUNCTION(ValueType x) { return ValueType{6} * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // initialize vectors
    auto rhs = vec::create(exec, gko::dim<2>(discretization_points, 1));
    generate_rhs(f, u0, u1, rhs.get());
    auto u = vec::create(exec, gko::dim<2>(discretization_points, 1));
    u->fill(0.0);

    // initialize the stencil matrix
    auto A = share(mtx::create(
        exec, gko::dim<2>{discretization_points, discretization_points}));
    generate_stencil_matrix(A.get());

    const RealValueType reduction_factor{1e-7};
    // Generate solver and solve the system
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(discretization_points),
            gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(
                reduction_factor))
        .with_preconditioner(bj::build())
        .on(exec)
        ->generate(A)
        ->apply(rhs, u);

    std::cout << "\nSolve complete."
              << "\nThe average relative error is "
              << calculate_error(discretization_points, u.get(), correct_u) /
                     discretization_points
              << std::endl;
}
