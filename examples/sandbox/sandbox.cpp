/*#include <ginkgo/ginkgo.hpp>

#include <iostream>
#include "core/factorization/symbolic.hpp"
#include "ginkgo/core/base/executor.hpp"
#include "ginkgo/core/matrix/csr.hpp"

using gko::size_type;
using gko::zero;


template <typename NonzeroIterator>
auto get_next_value(NonzeroIterator& it, const NonzeroIterator& end,
                    size_type next_row, size_type next_col) ->
    typename std::decay<decltype(it->value)>::type
{
    if (it != end && it->row == next_row && it->column == next_col) {
        return (it++)->value;
    } else {
        return zero<typename std::decay<decltype(it->value)>::type>();
    }
}

template <typename Ostream, typename MatrixData1, typename MatrixData2>
void print_sparsity_pattern(Ostream& os, const MatrixData1& first,
                            const MatrixData2& second)
{
    auto first_it = first.nonzeros.begin();
    auto second_it = second.nonzeros.begin();
    os << ' ';
    for (size_type col = 0; col < first.size[1]; col++) {
        os << (col % 10);
    }
    os << '\n';
    for (size_type row = 0; row < first.size[0]; row++) {
        os << (row % 10);
        for (size_type col = 0; col < first.size[1]; col++) {
            const auto has_first =
                get_next_value(first_it, end(first.nonzeros), row, col) !=
                zero<typename MatrixData1::value_type>();
            const auto has_second =
                get_next_value(second_it, end(second.nonzeros), row, col) !=
                zero<typename MatrixData2::value_type>();
            if (has_first) {
                if (has_second) {
                    os << '+';
                } else {
                    os << '|';
                }
            } else {
                if (has_second) {
                    os << '-';
                } else {
                    os << ' ';
                }
            }
        }
        os << (row % 10) << '\n';
    }
    os << ' ';
    for (size_type col = 0; col < first.size[1]; col++) {
        os << (col % 10);
    }
    os << "\n'|' is first, '-' is second, '+' is both, ' ' is none\n";
}*/


int main(int argc, char* argv[])
{
    /*auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::share(gko::read<gko::matrix::Csr<>>(std::cin, exec));
    auto perm = gko::reorder::Amd<int>::build().on(exec)->generate(mtx);
    auto arr = gko::make_array_view(exec, perm->get_size()[0],
                                    perm->get_permutation());
    gko::matrix_data<> data;
    auto reorder_mtx = gko::as<gko::matrix::Csr<>>(mtx->permute(&arr));
    reorder_mtx->sort_by_column_index();
    reorder_mtx->write(data);
    gko::write_raw(std::cout, data, gko::layout_type::coordinate);
    // print_sparsity_pattern(std::cout, data, data);
    std::unique_ptr<gko::matrix::Csr<>> orig_factors;
    std::unique_ptr<gko::matrix::Csr<>> reorder_factors;
    std::unique_ptr<gko::factorization::elimination_forest<int>> forest;
    gko::factorization::symbolic_cholesky(mtx.get(), orig_factors, forest);
    gko::factorization::symbolic_cholesky(reorder_mtx.get(), reorder_factors,
                                          forest);
    orig_factors->write(data);
    // print_sparsity_pattern(std::cout, data, data);
    reorder_factors->write(data);
    // print_sparsity_pattern(std::cout, data, data);
    std::cout << orig_factors->get_num_stored_elements() << ' '
              << reorder_factors->get_num_stored_elements() << '\n';*/
}
