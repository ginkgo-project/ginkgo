// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>
#include <iomanip>
#include <iostream>


// LU factorization implementation using Ginkgo ranges
// For simplicity, we only consider square matrices, and no pivoting.
template <typename Accessor>
void factorize(const gko::range<Accessor>& A)
// note: const means that the range (i.e. the data handler) is constant,
//       not that the underlying data is constant!
{
    using gko::span;
    assert(A.length(0) == A.length(1));
    for (gko::size_type i = 0; i < A.length(0) - 1; ++i) {
        const auto trail = span{i + 1, A.length(0)};
        // note: neither of the lines below need additional memory to store
        //       intermediate arrays, all computation is done at the point of
        //       assignment
        A(trail, i) = A(trail, i) / A(i, i);
        // caveat: operator * is element-wise multiplication, mmul is matrix
        //         multiplication
        A(trail, trail) = A(trail, trail) - mmul(A(trail, i), A(i, trail));
    }
}


// a utility function for printing the factorization on screen
template <typename Accessor>
void print_lu(const gko::range<Accessor>& A)
{
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "L = [";
    for (int i = 0; i < A.length(0); ++i) {
        std::cout << "\n  ";
        for (int j = 0; j < A.length(1); ++j) {
            std::cout << (i > j ? A(i, j) : (i == j) * 1.) << " ";
        }
    }
    std::cout << "\n]\n\nU = [";
    for (int i = 0; i < A.length(0); ++i) {
        std::cout << "\n  ";
        for (int j = 0; j < A.length(1); ++j) {
            std::cout << (i <= j ? A(i, j) : 0.) << " ";
        }
    }
    std::cout << "\n]" << std::endl;
}


int main(int argc, char* argv[])
{
    using ValueType = double;
    using IndexType = int;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Create some test data, add some padding just to demonstrate how to use it
    // with ranges.
    // clang-format off
    ValueType data[] = {
        2.,  4.,  5., -1.0,
        4., 11., 12., -1.0,
        6., 24., 24., -1.0
    };
    // clang-format on

    // Create a 3-by-3 range, with a 2D row-major accessor using data as the
    // underlying storage. Set the stride (a.k.a. "LDA") to 4.
    auto A =
        gko::range<gko::accessor::row_major<ValueType, 2>>(data, 3u, 3u, 4u);

    // use the LU factorization routine defined above to factorize the matrix
    factorize(A);

    // print the factorization on screen
    print_lu(A);
}
