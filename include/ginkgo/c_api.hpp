/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef C_API_H
#define C_API_H

#include <stddef.h>
#include <stdio.h>

/* ----------------------------------------------------------------------
 * Data type constants in GINKGO
 * ---------------------------------------------------------------------- */
enum _GKO_DATATYPE_CONST {
    GKO_NONE = -1,
    GKO_SHORT = 0,
    GKO_INT = 1,
    GKO_LONG_LONG = 2,
    GKO_FLOAT = 3,
    GKO_DOUBLE = 4,
    GKO_COMPLEX_FLOAT = 5,
    GKO_COMPLEX_DOUBLE = 6,
};


/* ----------------------------------------------------------------------
 * MACROS for generating structs for wrapping
 * ---------------------------------------------------------------------- */

/**
 * @brief A build instruction for defining gko::array<T>, simplifying its
 * construction by removing the repetitive typing of array's name.
 *
 * @param _ctype  Type name of the element type in C
 *
 * @param _cpptype  Type name of the element type in C++
 *
 * @param _name  Name of the datatype of the array
 *
 */
#define DEFINE_ARRAY_OVERLOAD(_ctype, _cpptype, _name)                         \
    struct gko_array_##_name##_st {                                            \
        gko::array<_cpptype> arr;                                              \
    };                                                                         \
                                                                               \
    typedef gko_array_##_name##_st* gko_array_##_name;                         \
                                                                               \
    gko_array_##_name ginkgo_array_##_name##_create(gko_executor exec_st_ptr,  \
                                                    size_t size)               \
    {                                                                          \
        return new gko_array_##_name##_st{                                     \
            gko::array<_cpptype>{exec_st_ptr->shared_ptr, size}};              \
    }                                                                          \
                                                                               \
    gko_array_##_name ginkgo_array_##_name##_create_view(                      \
        gko_executor exec_st_ptr, size_t size, _ctype* data_ptr)               \
    {                                                                          \
        return new gko_array_##_name##_st{gko::make_array_view(                \
            exec_st_ptr->shared_ptr, size, static_cast<_cpptype*>(data_ptr))}; \
    }                                                                          \
                                                                               \
    void ginkgo_array_##_name##_delete(gko_array_##_name array_st_ptr)         \
    {                                                                          \
        delete array_st_ptr;                                                   \
    }                                                                          \
                                                                               \
    size_t ginkgo_array_##_name##_get_num_elems(                               \
        gko_array_##_name array_st_ptr)                                        \
    {                                                                          \
        return (*array_st_ptr).arr.get_num_elems();                            \
    }

/**
 * @brief A build instruction for declaring gko::array<T> in the C API header
 * file
 */
#define DECLARE_ARRAY_OVERLOAD(_ctype, _cpptype, _name)                       \
    struct gko_array_##_name##_st;                                            \
    typedef struct gko_array_##_name##_st* gko_array_##_name;                 \
    gko_array_##_name ginkgo_array_##_name##_create(gko_executor exec_st_ptr, \
                                                    size_t size);             \
    gko_array_##_name ginkgo_array_##_name##_create_view(                     \
        gko_executor exec_st_ptr, size_t size, _ctype* data_ptr);             \
    void ginkgo_array_##_name##_delete(gko_array_##_name array_st_ptr);       \
    size_t ginkgo_array_##_name##_get_num_elems(gko_array_##_name array_st_ptr);


/**
 * @brief A build instruction for defining gko::matrix::Dense<T>, simplifying
 * its construction by removing the repetitive typing of array's name.
 *
 * @param _ctype  Type name of the element type in C
 *
 * @param _cpptype  Type name of the element type in C++
 *
 * @param _name  Name of the datatype of the dense matrix
 *
 */

#define DEFINE_DENSE_OVERLOAD(_ctype, _cpptype, _name)                    \
    struct gko_matrix_dense_##_name##_st {                                \
        std::shared_ptr<gko::matrix::Dense<_cpptype>> mat;                \
    };                                                                    \
                                                                          \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create(        \
        gko_executor exec, gko_dim2_st size)                              \
    {                                                                     \
        return new gko_matrix_dense_##_name##_st{                         \
            gko::matrix::Dense<_cpptype>::create(                         \
                (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols})};  \
    }                                                                     \
                                                                          \
    void ginkgo_matrix_dense_##_name##_delete(                            \
        gko_matrix_dense_##_name mat_st_ptr)                              \
    {                                                                     \
        delete mat_st_ptr;                                                \
    }                                                                     \
                                                                          \
    void ginkgo_matrix_dense_##_name##_fill(                              \
        gko_matrix_dense_##_name mat_st_ptr, _ctype value)                \
    {                                                                     \
        (*mat_st_ptr).mat->fill(value);                                   \
    }                                                                     \
                                                                          \
    _ctype ginkgo_matrix_dense_##_name##_at(                              \
        gko_matrix_dense_##_name mat_st_ptr, size_t row, size_t col)      \
    {                                                                     \
        return (*mat_st_ptr).mat->at(row, col);                           \
    }                                                                     \
                                                                          \
    gko_dim2_st ginkgo_matrix_dense_##_name##_get_size(                   \
        gko_matrix_dense_##_name mat_st_ptr)                              \
    {                                                                     \
        auto dim = (*mat_st_ptr).mat->get_size();                         \
        return gko_dim2_st{dim[0], dim[1]};                               \
    }                                                                     \
                                                                          \
    size_t ginkgo_matrix_dense_##_name##_get_num_stored_elements(         \
        gko_matrix_dense_##_name mat_st_ptr)                              \
    {                                                                     \
        return (*mat_st_ptr).mat->get_num_stored_elements();              \
    }                                                                     \
                                                                          \
    size_t ginkgo_matrix_dense_##_name##_get_stride(                      \
        gko_matrix_dense_##_name mat_st_ptr)                              \
    {                                                                     \
        return (*mat_st_ptr).mat->get_stride();                           \
    }                                                                     \
                                                                          \
    void ginkgo_matrix_dense_##_name##_compute_dot(                       \
        gko_matrix_dense_##_name mat_st_ptr1,                             \
        gko_matrix_dense_##_name mat_st_ptr2,                             \
        gko_matrix_dense_##_name mat_st_ptr_res)                          \
    {                                                                     \
        (*mat_st_ptr1)                                                    \
            .mat->compute_dot((*mat_st_ptr2).mat, (*mat_st_ptr_res).mat); \
    }                                                                     \
    void ginkgo_matrix_dense_##_name##_compute_norm1(                     \
        gko_matrix_dense_##_name mat_st_ptr1,                             \
        gko_matrix_dense_##_name mat_st_ptr2)                             \
    {                                                                     \
        (*mat_st_ptr1).mat->compute_norm1((*mat_st_ptr2).mat);            \
    }                                                                     \
                                                                          \
    void ginkgo_matrix_dense_##_name##_compute_norm2(                     \
        gko_matrix_dense_##_name mat_st_ptr1,                             \
        gko_matrix_dense_##_name mat_st_ptr2)                             \
    {                                                                     \
        (*mat_st_ptr1).mat->compute_norm2((*mat_st_ptr2).mat);            \
    }                                                                     \
                                                                          \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_read(          \
        const char* str_ptr, gko_executor exec)                           \
    {                                                                     \
        std::string filename(str_ptr);                                    \
        std::ifstream ifs(filename, std::ifstream::in);                   \
                                                                          \
        return new gko_matrix_dense_##_name##_st{                         \
            gko::read<gko::matrix::Dense<_cpptype>>(std::move(ifs),       \
                                                    (*exec).shared_ptr)}; \
    }                                                                     \
                                                                          \
    char* ginkgo_matrix_dense_##_name##_write_mtx(                        \
        gko_matrix_dense_##_name mat_st_ptr)                              \
    {                                                                     \
        auto cout_buff = std::cout.rdbuf();                               \
                                                                          \
        std::ostringstream local;                                         \
        std::cout.rdbuf(local.rdbuf());                                   \
        gko::write(std::cout, (*mat_st_ptr).mat);                         \
                                                                          \
        std::cout.rdbuf(cout_buff);                                       \
                                                                          \
        std::string str = local.str();                                    \
        char* cstr = new char[str.length() + 1];                          \
        std::strcpy(cstr, str.c_str());                                   \
        return cstr;                                                      \
    }

/**
 * @brief A build instruction for declaring gko::matrix::Dense<T> in the C API
 * header file
 */
#define DECLARE_DENSE_OVERLOAD(_ctype, _cpptype, _name)                     \
    struct gko_matrix_dense_##_name##_st;                                   \
    typedef struct gko_matrix_dense_##_name##_st* gko_matrix_dense_##_name; \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create(          \
        gko_executor exec, gko_dim2_st size);                               \
    void ginkgo_matrix_dense_##_name##_delete(                              \
        gko_matrix_dense_##_name mat_st_ptr);                               \
    void ginkgo_matrix_dense_##_name##_fill(                                \
        gko_matrix_dense_##_name mat_st_ptr, _ctype value);                 \
    _ctype ginkgo_matrix_dense_##_name##_at(                                \
        gko_matrix_dense_##_name mat_st_ptr, size_t row, size_t col);       \
    gko_dim2_st ginkgo_matrix_dense_##_name##_get_size(                     \
        gko_matrix_dense_##_name mat_st_ptr);                               \
    size_t ginkgo_matrix_dense_##_name##_get_num_stored_elements(           \
        gko_matrix_dense_##_name mat_st_ptr);                               \
    size_t ginkgo_matrix_dense_##_name##_get_stride(                        \
        gko_matrix_dense_##_name mat_st_ptr);                               \
    void ginkgo_matrix_dense_##_name##_compute_dot(                         \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2,                               \
        gko_matrix_dense_##_name mat_st_ptr_res);                           \
    void ginkgo_matrix_dense_##_name##_compute_norm1(                       \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2);                              \
    void ginkgo_matrix_dense_##_name##_compute_norm2(                       \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2);                              \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_read(            \
        const char* str_ptr, gko_executor exec);                            \
    char* ginkgo_matrix_dense_##_name##_write_mtx(                          \
        gko_matrix_dense_##_name mat_st_ptr);

#ifdef __cplusplus
extern "C" {
#endif


/* ----------------------------------------------------------------------
 * C memory management
 * ---------------------------------------------------------------------- */
void c_char_ptr_free(char* ptr);

/* ----------------------------------------------------------------------
 * Library functions for some basic types in GINKGO
 * ---------------------------------------------------------------------- */
/**
 * @brief Struct implements the gko::dim<2> type
 *
 */
typedef struct {
    size_t rows;
    size_t cols;
} gko_dim2_st;

/**
 * @brief Allocates memory for a C-based reimplementation of the gko::dim<2>
 * type
 *
 * @param rows First dimension
 * @param cols Second dimension
 * @return gko_dim2_st C struct that contains members of the gko::dim<2> type
 */
gko_dim2_st ginkgo_dim2_create(size_t rows, size_t cols);

/**
 * @brief Obtains the value of the first element of a gko::dim<2> type
 *
 * @param dim An object of gko_dim2_st type
 * @return size_t First dimension
 */
size_t ginkgo_dim2_rows_get(gko_dim2_st dim);

/**
 * @brief Obtains the value of the second element of a gko::dim<2> type
 *
 * @param dim An object of gko_dim2_st type
 * @return size_t Second dimension
 */
size_t ginkgo_dim2_cols_get(gko_dim2_st dim);


/* ----------------------------------------------------------------------
 * Library functions and structs for creating executors in GINKGO
 * ---------------------------------------------------------------------- */

/**
 * @brief Struct containing the shared pointer to a ginkgo executor
 *
 */
struct gko_executor_st;

/**
 * @brief Type of the pointer to the wrapped `gko_executor_st` struct
 *
 */
typedef struct gko_executor_st* gko_executor;

/**
 * @brief Deallocates memory for an executor on targeted device.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the executor to be
 * deleted
 */
void ginkgo_executor_delete(gko_executor exec_st_ptr);

gko_executor ginkgo_executor_omp_create();
gko_executor ginkgo_executor_cuda_create();
// gko_executor ginkgo_executor_hip_create();
// gko_executor ginkgo_executor_dpcpp_create();
gko_executor ginkgo_executor_reference_create();

/* ----------------------------------------------------------------------
 * Library functions for creating arrays and array operations in GINKGO
 * ---------------------------------------------------------------------- */
DECLARE_ARRAY_OVERLOAD(short, short, i16);
DECLARE_ARRAY_OVERLOAD(int, int, i32);
DECLARE_ARRAY_OVERLOAD(long long, long long, i64);
DECLARE_ARRAY_OVERLOAD(float, float, f32);
DECLARE_ARRAY_OVERLOAD(double, double, f64);
// DECLARE_ARRAY_OVERLOAD(float complex, std::complex<float>, cf32);
// DECLARE_ARRAY_OVERLOAD(double complex, std::complex<double>, cf64);


/* ----------------------------------------------------------------------
 * Library functions for creating matrices and matrix operations in GINKGO
 * ---------------------------------------------------------------------- */
DECLARE_DENSE_OVERLOAD(float, float, f32);
DECLARE_DENSE_OVERLOAD(double, double, f64);
// DECLARE_DENSE_OVERLOAD(float _Complex, std::complex<float>, cf32);
// DECLARE_DENSE_OVERLOAD(double _Complex, std::complex<double>, cf64);
// DECLARE_DENSE_OVERLOAD(short, short, i16);
// DECLARE_DENSE_OVERLOAD(int, int, i32);
// DECLARE_DENSE_OVERLOAD(long long, long long, i64);

struct gko_matrix_csr_f32_i32_st;
typedef struct gko_matrix_csr_f32_i32_st* gko_matrix_csr_f32_i32;
gko_matrix_csr_f32_i32 ginkgo_matrix_csr_f32_i32_create(gko_executor exec,
                                                        gko_dim2_st size,
                                                        size_t nnz);
void ginkgo_matrix_csr_f32_i32_delete(gko_matrix_csr_f32_i32 mat_st_ptr);
gko_matrix_csr_f32_i32 ginkgo_matrix_csr_f32_i32_read(const char* str_ptr,
                                                      gko_executor exec);
size_t ginkgo_matrix_csr_f32_i32_get_num_stored_elements(
    gko_matrix_csr_f32_i32 mat_st_ptr);
size_t ginkgo_matrix_csr_f32_i32_get_num_srow_elements(
    gko_matrix_csr_f32_i32 mat_st_ptr);
gko_dim2_st ginkgo_matrix_csr_f32_i32_get_size(
    gko_matrix_csr_f32_i32 mat_st_ptr);
const float* ginkgo_matrix_csr_f32_i32_get_const_values(
    gko_matrix_csr_f32_i32 mat_st_ptr);
const int* ginkgo_matrix_csr_f32_i32_get_const_col_idxs(
    gko_matrix_csr_f32_i32 mat_st_ptr);
const int* ginkgo_matrix_csr_f32_i32_get_const_row_ptrs(
    gko_matrix_csr_f32_i32 mat_st_ptr);
const int* ginkgo_matrix_csr_f32_i32_get_const_srow(
    gko_matrix_csr_f32_i32 mat_st_ptr);

/**
 * @brief Performs an SpMM product
 *
 * @param mat_st_ptr
 * @param alpha
 * @param x
 * @param beta
 * @param y
 */
void ginkgo_matrix_csr_f32_i32_apply(gko_matrix_csr_f32_i32 mat_st_ptr,
                                     gko_matrix_dense_f32 alpha,
                                     gko_matrix_dense_f32 x,
                                     gko_matrix_dense_f32 beta,
                                     gko_matrix_dense_f32 y);


/* ----------------------------------------------------------------------
 * Library functions for BLAS linop in GINKGO
 * ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
 * Library functions for iterative solvers in GINKGO
 * ---------------------------------------------------------------------- */
void ginkgo_solver_cg_solve(gko_executor exec_st_ptr,
                            gko_matrix_csr_f32_i32 A_st_ptr,
                            gko_matrix_dense_f32 b_st_ptr,
                            gko_matrix_dense_f32 x_st_ptr, int maxiter,
                            double reduction);

/* ----------------------------------------------------------------------
 * Library functions for retrieving configuration information in GINKGO
 * ---------------------------------------------------------------------- */

/**
 * @brief This function is a wrapper for obtaining the version of the ginkgo
 * library
 *
 */
void ginkgo_version_get();


#ifdef __cplusplus
}
#endif

#endif /* C_API_H */