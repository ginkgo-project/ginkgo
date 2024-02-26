// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/mtx_io.hpp>


#include <cstring>
#include <sstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


TEST(MtxReader, ReadsDenseDoubleRealMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseDoubleRealMtxWith64Index)
{
    using tpl = gko::matrix_data<double, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto data = gko::read_raw<double, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseFloatIntegerMtx)
{
    using tpl = gko::matrix_data<float, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array integer general\n"
        "2 3\n"
        "1\n"
        "0\n"
        "3\n"
        "5\n"
        "2\n"
        "0\n");

    auto data = gko::read_raw<float, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseFloatIntegerMtxWith64Index)
{
    using tpl = gko::matrix_data<float, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array integer general\n"
        "2 3\n"
        "1\n"
        "0\n"
        "3\n"
        "5\n"
        "2\n"
        "0\n");

    auto data = gko::read_raw<float, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseComplexDoubleMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexDoubleMtxWith64Index)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexFloatMtx)
{
    using cpx = std::complex<float>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexFloatMtxWith64Index)
{
    using cpx = std::complex<float>;
    using tpl = gko::matrix_data<cpx, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsSparseRealMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 3 4\n"
        "1 1 1.0\n"
        "2 2 5.0\n"
        "1 2 3.0\n"
        "1 3 2.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 1, 5.0));
}


TEST(MtxReader, ReadsSparseRealSymmetricMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 3 6.0\n"
        "3 1 3.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 2.0));
    ASSERT_EQ(v[2], tpl(0, 2, 3.0));
    ASSERT_EQ(v[3], tpl(1, 0, 2.0));
    ASSERT_EQ(v[4], tpl(2, 0, 3.0));
    ASSERT_EQ(v[5], tpl(2, 2, 6.0));
}


TEST(MtxReader, ReadsSparseRealSkewSymmetricMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n"
        "3 3 2\n"
        "2 1 2.0\n"
        "3 1 3.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, -2.0));
    ASSERT_EQ(v[1], tpl(0, 2, -3.0));
    ASSERT_EQ(v[2], tpl(1, 0, 2.0));
    ASSERT_EQ(v[3], tpl(2, 0, 3.0));
}


TEST(MtxReader, ReadsSparseRealSkewSymmetricMtxWithExplicitDiagonal)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n"
        "3 3 3\n"
        "1 1 0.0\n"
        "2 1 2.0\n"
        "3 1 3.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 0.0));
    ASSERT_EQ(v[1], tpl(0, 1, -2.0));
    ASSERT_EQ(v[2], tpl(0, 2, -3.0));
    ASSERT_EQ(v[3], tpl(1, 0, 2.0));
    ASSERT_EQ(v[4], tpl(2, 0, 3.0));
}


TEST(MtxReader, ReadsSparsePatternMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate pattern general\n"
        "2 3 4\n"
        "1 1\n"
        "2 2\n"
        "1 2\n"
        "1 3\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 1.0));
    ASSERT_EQ(v[2], tpl(0, 2, 1.0));
    ASSERT_EQ(v[3], tpl(1, 1, 1.0));
}


TEST(MtxReader, ReadsSparseComplexMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex general\n"
        "2 3 4\n"
        "1 1 1.0 2.0\n"
        "2 2 5.0 3.0\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 1, cpx(5.0, 3.0)));
}


TEST(MtxReader, ReadsSparseComplexHermitianMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex hermitian\n"
        "2 3 2\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[1], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[2], tpl(1, 0, cpx(3.0, -1.0)));
    ASSERT_EQ(v[3], tpl(2, 0, cpx(2.0, -4.0)));
}


std::array<gko::uint64, 20> build_binary_complex_data()
{
    gko::uint64 int_val{};
    gko::uint64 neg_int_val{};
    double dbl_val = 2.5;
    double neg_dbl_val = -2.5;
    std::memcpy(&int_val, &dbl_val, sizeof(double));
    std::memcpy(&neg_int_val, &neg_dbl_val, sizeof(double));
    constexpr gko::uint64 shift = 256;
    // note: the following data is not sorted!
    std::array<gko::uint64, 20> data{
        'G' + shift *
                  ('I' +
                   shift *
                       ('N' +
                        shift *
                            ('K' +
                             shift * ('G' +
                                      shift * ('O' +
                                               shift * ('Z' + shift * 'L')))))),
        64,       // num_rows
        32,       // num_cols
        4,        // num_entries
        0,        // row
        1,        // col
        0,        // real val
        int_val,  // imag val
        1,        // row
        1,        // col
        int_val,
        neg_int_val,
        16,  // row
        25,  // col
        0,
        0,
        4,  // row
        2,  // col
        0,
        neg_int_val};
    return data;
}


std::array<gko::uint64, 16> build_binary_real_data()
{
    gko::uint64 int_val{};
    gko::uint64 neg_int_val{};
    double dbl_val = 2.5;
    double neg_dbl_val = -2.5;
    std::memcpy(&int_val, &dbl_val, sizeof(double));
    std::memcpy(&neg_int_val, &neg_dbl_val, sizeof(double));
    constexpr gko::uint64 shift = 256;
    // note: the following data is not sorted!
    std::array<gko::uint64, 16> data{
        'G' + shift *
                  ('I' +
                   shift *
                       ('N' +
                        shift *
                            ('K' +
                             shift * ('G' +
                                      shift * ('O' +
                                               shift * ('D' + shift * 'L')))))),
        64,  // num_rows
        32,  // num_cols
        4,   // num_entries
        1,   // row
        1,   // col
        int_val,
        0,  // row
        1,  // col
        0,  // val
        4,  // row
        2,  // col
        neg_int_val,
        16,  // row
        25,  // col
        0};
    return data;
}


TEST(MtxReader, ReadsBinary)
{
    auto raw_data = build_binary_real_data();
    auto test_read = [&](auto mtx_data) {
        SCOPED_TRACE(gko::name_demangling::get_static_type(mtx_data));
        using value_type =
            typename std::decay_t<decltype(mtx_data)>::value_type;
        using index_type =
            typename std::decay_t<decltype(mtx_data)>::index_type;
        std::stringstream ss{
            std::string{reinterpret_cast<char*>(raw_data.data()),
                        raw_data.size() * sizeof(gko::uint64)}};

        auto data = gko::read_binary_raw<value_type, index_type>(ss);

        ASSERT_EQ(data.size, gko::dim<2>(64, 32));
        ASSERT_EQ(data.nonzeros.size(), 4);
        ASSERT_EQ(data.nonzeros[0].row, 0);
        ASSERT_EQ(data.nonzeros[1].row, 1);
        ASSERT_EQ(data.nonzeros[2].row, 4);
        ASSERT_EQ(data.nonzeros[3].row, 16);
        ASSERT_EQ(data.nonzeros[0].column, 1);
        ASSERT_EQ(data.nonzeros[1].column, 1);
        ASSERT_EQ(data.nonzeros[2].column, 2);
        ASSERT_EQ(data.nonzeros[3].column, 25);
        ASSERT_EQ(data.nonzeros[0].value, value_type{0.0});
        ASSERT_EQ(data.nonzeros[1].value, value_type{2.5});
        ASSERT_EQ(data.nonzeros[2].value, value_type{-2.5});
        ASSERT_EQ(data.nonzeros[3].value, value_type{0.0});
    };

    test_read(gko::matrix_data<float, gko::int32>{});
    test_read(gko::matrix_data<double, gko::int32>{});
    test_read(gko::matrix_data<std::complex<float>, gko::int32>{});
    test_read(gko::matrix_data<std::complex<double>, gko::int32>{});
    test_read(gko::matrix_data<float, gko::int64>{});
    test_read(gko::matrix_data<double, gko::int64>{});
    test_read(gko::matrix_data<std::complex<float>, gko::int64>{});
    test_read(gko::matrix_data<std::complex<double>, gko::int64>{});
}


TEST(MtxReader, ReadsComplexBinary)
{
    auto raw_data = build_binary_complex_data();
    auto test_read = [&](auto mtx_data) {
        SCOPED_TRACE(gko::name_demangling::get_static_type(mtx_data));
        using value_type =
            typename std::decay_t<decltype(mtx_data)>::value_type;
        using index_type =
            typename std::decay_t<decltype(mtx_data)>::index_type;
        std::stringstream ss{
            std::string{reinterpret_cast<char*>(raw_data.data()),
                        raw_data.size() * sizeof(gko::uint64)}};
        auto data = gko::read_binary_raw<value_type, index_type>(ss);

        ASSERT_EQ(data.size, gko::dim<2>(64, 32));
        ASSERT_EQ(data.nonzeros.size(), 4);
        ASSERT_EQ(data.nonzeros[0].row, 0);
        ASSERT_EQ(data.nonzeros[1].row, 1);
        ASSERT_EQ(data.nonzeros[2].row, 4);
        ASSERT_EQ(data.nonzeros[3].row, 16);
        ASSERT_EQ(data.nonzeros[0].column, 1);
        ASSERT_EQ(data.nonzeros[1].column, 1);
        ASSERT_EQ(data.nonzeros[2].column, 2);
        ASSERT_EQ(data.nonzeros[3].column, 25);
        ASSERT_EQ(data.nonzeros[0].value, value_type(0.0, 2.5));
        ASSERT_EQ(data.nonzeros[1].value, value_type(2.5, -2.5));
        ASSERT_EQ(data.nonzeros[2].value, value_type(0.0, -2.5));
        ASSERT_EQ(data.nonzeros[3].value, value_type(0.0, 0.0));
    };

    auto test_read_fail = [&](auto mtx_data) {
        SCOPED_TRACE(gko::name_demangling::get_static_type(mtx_data));
        using value_type =
            typename std::decay_t<decltype(mtx_data)>::value_type;
        using index_type =
            typename std::decay_t<decltype(mtx_data)>::index_type;
        std::stringstream ss{
            std::string{reinterpret_cast<char*>(raw_data.data()),
                        raw_data.size() * sizeof(gko::uint64)}};

        ASSERT_THROW((gko::read_binary_raw<value_type, index_type>(ss)),
                     gko::StreamError);
    };

    test_read_fail(gko::matrix_data<float, gko::int32>{});
    test_read_fail(gko::matrix_data<double, gko::int32>{});
    test_read(gko::matrix_data<std::complex<float>, gko::int32>{});
    test_read(gko::matrix_data<std::complex<double>, gko::int32>{});
    test_read_fail(gko::matrix_data<float, gko::int64>{});
    test_read_fail(gko::matrix_data<double, gko::int64>{});
    test_read(gko::matrix_data<std::complex<float>, gko::int64>{});
    test_read(gko::matrix_data<std::complex<double>, gko::int64>{});
}


TEST(MtxReader, ReadsGenericBinary)
{
    auto raw_data = build_binary_real_data();
    std::stringstream ss{std::string{reinterpret_cast<char*>(raw_data.data()),
                                     raw_data.size() * sizeof(gko::uint64)}};

    auto data = gko::read_generic_raw<double, int>(ss);

    ASSERT_EQ(data.size, gko::dim<2>(64, 32));
    ASSERT_EQ(data.nonzeros.size(), 4);
    ASSERT_EQ(data.nonzeros[0].row, 0);
    ASSERT_EQ(data.nonzeros[1].row, 1);
    ASSERT_EQ(data.nonzeros[2].row, 4);
    ASSERT_EQ(data.nonzeros[3].row, 16);
    ASSERT_EQ(data.nonzeros[0].column, 1);
    ASSERT_EQ(data.nonzeros[1].column, 1);
    ASSERT_EQ(data.nonzeros[2].column, 2);
    ASSERT_EQ(data.nonzeros[3].column, 25);
    ASSERT_EQ(data.nonzeros[0].value, 0.0);
    ASSERT_EQ(data.nonzeros[1].value, 2.5);
    ASSERT_EQ(data.nonzeros[2].value, -2.5);
    ASSERT_EQ(data.nonzeros[3].value, 0.0);
}


TEST(MtxReader, ReadsGenericMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 3 4\n"
        "1 1 1.0\n"
        "2 2 5.0\n"
        "1 2 3.0\n"
        "1 3 2.0\n");

    auto data = gko::read_generic_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 1, 5.0));
}


TEST(MtxReader, FailsWhenReadingSparseComplexMtxToRealMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex general\n"
        "2 3 4\n"
        "1 1 1.0 2.0\n"
        "2 2 5.0 3.0\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    ASSERT_THROW((gko::read_raw<double, gko::int32>(iss)), gko::StreamError);
}


TEST(MatrixData, WritesDoubleRealMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<double, gko::int32> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "3 2\n"
              "1\n"
              "2.1\n"
              "3\n"
              "2\n"
              "0\n"
              "3.2\n");
}


TEST(MatrixData, WritesFloatRealMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<float, gko::int32> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n"
              "3 2 5\n"
              "1 1 1\n"
              "1 2 2\n"
              "2 1 2.1\n"
              "3 1 3\n"
              "3 2 3.2\n");
}


TEST(MatrixData, WritesDoubleRealMatrixToMatrixMarketArrayWith64Index)
{
    // clang-format off
    gko::matrix_data<double, gko::int64> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "3 2\n"
              "1\n"
              "2.1\n"
              "3\n"
              "2\n"
              "0\n"
              "3.2\n");
}


TEST(MatrixData, WritesFloatRealMatrixToMatrixMarketCoordinateWith64Index)
{
    // clang-format off
    gko::matrix_data<float, gko::int64> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n"
              "3 2 5\n"
              "1 1 1\n"
              "1 2 2\n"
              "2 1 2.1\n"
              "3 1 3\n"
              "3 2 3.2\n");
}


TEST(MatrixData, WritesComplexDoubleMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<std::complex<double>, gko::int32> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "3 2\n"
              "1 0\n"
              "2.1 2.2\n"
              "0 3\n"
              "2 3.2\n"
              "0 0\n"
              "3.2 5.3\n");
}


TEST(MatrixData, WritesComplexFloatMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<std::complex<float>, gko::int32> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate complex general\n"
              "3 2 5\n"
              "1 1 1 0\n"
              "1 2 2 3.2\n"
              "2 1 2.1 2.2\n"
              "3 1 0 3\n"
              "3 2 3.2 5.3\n");
}


TEST(MatrixData, WritesComplexDoubleMatrixToMatrixMarketArrayWith64Index)
{
    // clang-format off
    gko::matrix_data<std::complex<double>, gko::int64> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "3 2\n"
              "1 0\n"
              "2.1 2.2\n"
              "0 3\n"
              "2 3.2\n"
              "0 0\n"
              "3.2 5.3\n");
}


TEST(MatrixData, WritesComplexFloatMatrixToMatrixMarketCoordinateWith64Index)
{
    // clang-format off
    gko::matrix_data<std::complex<float>, gko::int64> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate complex general\n"
              "3 2 5\n"
              "1 1 1 0\n"
              "1 2 2 3.2\n"
              "2 1 2.1 2.2\n"
              "3 1 0 3\n"
              "3 2 3.2 5.3\n");
}


TEST(MtxReader, WritesBinary)
{
    auto ref_data = build_binary_real_data();
    std::stringstream ss;
    gko::matrix_data<double, gko::int64> data;
    data.size = gko::dim<2>{64, 32};
    data.nonzeros.resize(4);
    data.nonzeros[0] = {1, 1, 2.5};
    data.nonzeros[1] = {0, 1, 0.0};
    data.nonzeros[2] = {4, 2, -2.5};
    data.nonzeros[3] = {16, 25, 0.0};

    gko::write_binary_raw(ss, data);

    ASSERT_EQ(ss.str(), std::string(reinterpret_cast<char*>(ref_data.data()),
                                    ref_data.size() * sizeof(gko::uint64)));
}


TEST(MtxReader, WritesComplexBinary)
{
    auto ref_data = build_binary_complex_data();
    std::stringstream ss;
    gko::matrix_data<std::complex<double>, gko::int64> data;
    data.size = gko::dim<2>{64, 32};
    data.nonzeros.resize(4);
    data.nonzeros[0] = {0, 1, {0.0, 2.5}};
    data.nonzeros[1] = {1, 1, {2.5, -2.5}};
    data.nonzeros[2] = {16, 25, {0.0, 0.0}};
    data.nonzeros[3] = {4, 2, {0.0, -2.5}};

    gko::write_binary_raw(ss, data);

    ASSERT_EQ(ss.str(), std::string(reinterpret_cast<char*>(ref_data.data()),
                                    ref_data.size() * sizeof(gko::uint64)));
}


template <typename ValueType, typename IndexType>
class DummyLinOp
    : public gko::EnableLinOp<DummyLinOp<ValueType, IndexType>>,
      public gko::EnableCreateMethod<DummyLinOp<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType>,
      public gko::WritableToMatrixData<ValueType, IndexType> {
    friend class gko::EnablePolymorphicObject<DummyLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<DummyLinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data& data) override { data_ = data; }

    void write(mat_data& data) const override { data = data_; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOp>(exec)
    {}

public:
    mat_data data_;
};


template <typename ValueIndexType>
class RealDummyLinOpTest : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(RealDummyLinOpTest, gko::test::RealValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(RealDummyLinOpTest, ReadsLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TYPED_TEST(RealDummyLinOpTest, ReadsGenericLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto lin_op = gko::read_generic<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TYPED_TEST(RealDummyLinOpTest, ReadsGenericLinOpFromBinaryStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    auto raw_data = build_binary_real_data();
    std::istringstream iss(std::string{reinterpret_cast<char*>(raw_data.data()),
                                       raw_data.size() * sizeof(gko::uint64)});

    auto lin_op = gko::read_generic<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(64, 32));
    ASSERT_EQ(data.nonzeros.size(), 4);
    ASSERT_EQ(data.nonzeros[0], tpl(0, 1, value_type{0.0}));
    ASSERT_EQ(data.nonzeros[1], tpl(1, 1, value_type{2.5}));
    ASSERT_EQ(data.nonzeros[2], tpl(4, 2, value_type{-2.5}));
    ASSERT_EQ(data.nonzeros[3], tpl(16, 25, value_type{0.0}));
}


TYPED_TEST(RealDummyLinOpTest, WritesLinOpToStreamArray)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lin_op, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "2 3\n"
              "1\n"
              "0\n"
              "3\n"
              "5\n"
              "2\n"
              "0\n");
}


TYPED_TEST(RealDummyLinOpTest, WritesLinOpToStreamCoordinate)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lin_op, gko::layout_type::coordinate);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n2 3 6\n1 1 1\n1 "
              "2 3\n1 3 2\n2 1 0\n2 2 5\n2 3 0\n");
}


TYPED_TEST(RealDummyLinOpTest, WritesLinOpToStreamDefault)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};
    std::ostringstream oss_const{};

    write(oss, lin_op);
    write(oss_const, std::unique_ptr<const DummyLinOp<value_type, index_type>>{
                         std::move(lin_op)});

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n2 3 6\n1 1 1\n1 "
              "2 3\n1 3 2\n2 1 0\n2 2 5\n2 3 0\n");
    ASSERT_EQ(oss_const.str(), oss.str());
}


TYPED_TEST(RealDummyLinOpTest, WritesAndReadsBinaryLinOpToStreamArray)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto ref = gko::ReferenceExecutor::create();
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(iss, ref);
    std::ostringstream oss{};

    gko::write_binary(oss, lin_op);
    std::istringstream iss2{oss.str()};
    auto lin_op2 =
        gko::read_binary<DummyLinOp<value_type, index_type>>(iss2, ref);

    ASSERT_EQ(lin_op->data_.size, lin_op2->data_.size);
    ASSERT_EQ(lin_op->data_.nonzeros, lin_op2->data_.nonzeros);
}


template <typename ValueIndexType>
class DenseTest : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<0, ValueIndexType>::type;
    using index_type = typename std::tuple_element<1, ValueIndexType>::type;
};

TYPED_TEST_SUITE(DenseTest, gko::test::RealValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(DenseTest, WritesToStreamDefault)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto lin_op = gko::read<gko::matrix::Dense<value_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lin_op);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "2 3\n"
              "1\n"
              "0\n"
              "3\n"
              "5\n"
              "2\n"
              "0\n");
}


template <typename ValueIndexType>
class ComplexDummyLinOpTest : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(ComplexDummyLinOpTest, gko::test::ComplexValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ComplexDummyLinOpTest, ReadsLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");

    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, value_type{1.0, 2.0}));
    ASSERT_EQ(v[1], tpl(0, 1, value_type{3.0, 4.0}));
    ASSERT_EQ(v[2], tpl(0, 2, value_type{2.0, 3.0}));
    ASSERT_EQ(v[3], tpl(1, 0, value_type{0.0, 0.0}));
    ASSERT_EQ(v[4], tpl(1, 1, value_type{5.0, 6.0}));
    ASSERT_EQ(v[5], tpl(1, 2, value_type{0.0, 0.0}));
}


TYPED_TEST(ComplexDummyLinOpTest, ReadsGenericLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");

    auto lin_op = gko::read_generic<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto& v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, value_type{1.0, 2.0}));
    ASSERT_EQ(v[1], tpl(0, 1, value_type{3.0, 4.0}));
    ASSERT_EQ(v[2], tpl(0, 2, value_type{2.0, 3.0}));
    ASSERT_EQ(v[3], tpl(1, 0, value_type{0.0, 0.0}));
    ASSERT_EQ(v[4], tpl(1, 1, value_type{5.0, 6.0}));
    ASSERT_EQ(v[5], tpl(1, 2, value_type{0.0, 0.0}));
}


TYPED_TEST(ComplexDummyLinOpTest, ReadsGenericLinOpFromBinaryStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto raw_data = build_binary_real_data();
    std::istringstream iss(std::string{reinterpret_cast<char*>(raw_data.data()),
                                       raw_data.size() * sizeof(gko::uint64)});

    auto lin_op = gko::read_generic<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto& data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(64, 32));
    ASSERT_EQ(data.nonzeros.size(), 4);
    ASSERT_EQ(data.nonzeros[0].row, 0);
    ASSERT_EQ(data.nonzeros[1].row, 1);
    ASSERT_EQ(data.nonzeros[2].row, 4);
    ASSERT_EQ(data.nonzeros[3].row, 16);
    ASSERT_EQ(data.nonzeros[0].column, 1);
    ASSERT_EQ(data.nonzeros[1].column, 1);
    ASSERT_EQ(data.nonzeros[2].column, 2);
    ASSERT_EQ(data.nonzeros[3].column, 25);
    ASSERT_EQ(data.nonzeros[0].value, value_type{0.0});
    ASSERT_EQ(data.nonzeros[1].value, value_type{2.5});
    ASSERT_EQ(data.nonzeros[2].value, value_type{-2.5});
    ASSERT_EQ(data.nonzeros[3].value, value_type{0.0});
}


TYPED_TEST(ComplexDummyLinOpTest, WritesLinOpToStreamArray)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lin_op, gko::layout_type::array);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "2 3\n"
              "1 2\n"
              "0 0\n"
              "3 4\n"
              "5 6\n"
              "2 3\n"
              "0 0\n");
}


TYPED_TEST(ComplexDummyLinOpTest, WritesAndReadsBinaryLinOpToStreamArray)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");
    auto ref = gko::ReferenceExecutor::create();
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(iss, ref);
    std::ostringstream oss{};

    gko::write_binary(oss, lin_op);
    std::istringstream iss2{oss.str()};
    auto lin_op2 =
        gko::read_binary<DummyLinOp<value_type, index_type>>(iss2, ref);

    ASSERT_EQ(lin_op->data_.size, lin_op2->data_.size);
    ASSERT_EQ(lin_op->data_.nonzeros, lin_op2->data_.nonzeros);
}


}  // namespace
