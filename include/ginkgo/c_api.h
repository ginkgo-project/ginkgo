// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_C_API_H
#define GKO_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ----------------------------------------------------------------------
 * MACROS for generating structs for wrapping
 * ---------------------------------------------------------------------- */

/**
 * @brief A build instruction for defining gko::array<T>, simplifying its
 * construction by removing the repetitive typing of array's name.
 *
 * @param _ctype  Type name of the element type in C
 * @param _cpptype  Type name of the element type in C++
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
    size_t ginkgo_array_##_name##_get_size(gko_array_##_name array_st_ptr)     \
    {                                                                          \
        return (*array_st_ptr).arr.get_size();                                 \
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
    size_t ginkgo_array_##_name##_get_size(gko_array_##_name array_st_ptr);


/**
 * @brief A build instruction for defining gko::matrix::Dense<T>, simplifying
 * its construction by removing the repetitive typing of array's name.
 *
 * @param _ctype  Type name of the element type in C
 * @param _cpptype  Type name of the element type in C++
 * @param _name  Name of the datatype of the dense matrix
 *
 */
#define DEFINE_DENSE_OVERLOAD(_ctype, _cpptype, _name)                      \
    struct gko_matrix_dense_##_name##_st {                                  \
        std::shared_ptr<gko::matrix::Dense<_cpptype>> mat;                  \
    };                                                                      \
                                                                            \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create(          \
        gko_executor exec, gko_dim2_st size)                                \
    {                                                                       \
        return new gko_matrix_dense_##_name##_st{                           \
            gko::matrix::Dense<_cpptype>::create(                           \
                (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols})};    \
    }                                                                       \
                                                                            \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create_view(     \
        gko_executor exec, gko_dim2_st size, _ctype* values, size_t stride) \
    {                                                                       \
        return new gko_matrix_dense_##_name##_st{                           \
            gko::matrix::Dense<_ctype>::create(                             \
                (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols},      \
                gko::array<_ctype>::view((*exec).shared_ptr, size.rows,     \
                                         values),                           \
                stride)};                                                   \
    }                                                                       \
                                                                            \
    void ginkgo_matrix_dense_##_name##_delete(                              \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        delete mat_st_ptr;                                                  \
    }                                                                       \
                                                                            \
    void ginkgo_matrix_dense_##_name##_fill(                                \
        gko_matrix_dense_##_name mat_st_ptr, _ctype value)                  \
    {                                                                       \
        (*mat_st_ptr).mat->fill(value);                                     \
    }                                                                       \
                                                                            \
    _ctype ginkgo_matrix_dense_##_name##_at(                                \
        gko_matrix_dense_##_name mat_st_ptr, size_t row, size_t col)        \
    {                                                                       \
        return (*mat_st_ptr).mat->at(row, col);                             \
    }                                                                       \
                                                                            \
    gko_dim2_st ginkgo_matrix_dense_##_name##_get_size(                     \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        auto dim = (*mat_st_ptr).mat->get_size();                           \
        return gko_dim2_st{dim[0], dim[1]};                                 \
    }                                                                       \
                                                                            \
    _ctype* ginkgo_matrix_dense_##_name##_get_values(                       \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        return (*mat_st_ptr).mat->get_values();                             \
    }                                                                       \
                                                                            \
    const _ctype* ginkgo_matrix_dense_##_name##_get_const_values(           \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        return (*mat_st_ptr).mat->get_const_values();                       \
    }                                                                       \
                                                                            \
    size_t ginkgo_matrix_dense_##_name##_get_num_stored_elements(           \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        return (*mat_st_ptr).mat->get_num_stored_elements();                \
    }                                                                       \
                                                                            \
    size_t ginkgo_matrix_dense_##_name##_get_stride(                        \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        return (*mat_st_ptr).mat->get_stride();                             \
    }                                                                       \
                                                                            \
    void ginkgo_matrix_dense_##_name##_compute_dot(                         \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2,                               \
        gko_matrix_dense_##_name mat_st_ptr_res)                            \
    {                                                                       \
        (*mat_st_ptr1)                                                      \
            .mat->compute_dot((*mat_st_ptr2).mat, (*mat_st_ptr_res).mat);   \
    }                                                                       \
    void ginkgo_matrix_dense_##_name##_compute_norm1(                       \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2)                               \
    {                                                                       \
        (*mat_st_ptr1).mat->compute_norm1((*mat_st_ptr2).mat);              \
    }                                                                       \
                                                                            \
    void ginkgo_matrix_dense_##_name##_compute_norm2(                       \
        gko_matrix_dense_##_name mat_st_ptr1,                               \
        gko_matrix_dense_##_name mat_st_ptr2)                               \
    {                                                                       \
        (*mat_st_ptr1).mat->compute_norm2((*mat_st_ptr2).mat);              \
    }                                                                       \
                                                                            \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_read(            \
        const char* str_ptr, gko_executor exec)                             \
    {                                                                       \
        std::string filename(str_ptr);                                      \
        std::ifstream ifs(filename, std::ifstream::in);                     \
                                                                            \
        return new gko_matrix_dense_##_name##_st{                           \
            gko::read<gko::matrix::Dense<_cpptype>>(std::move(ifs),         \
                                                    (*exec).shared_ptr)};   \
    }                                                                       \
                                                                            \
    char* ginkgo_matrix_dense_##_name##_write_mtx(                          \
        gko_matrix_dense_##_name mat_st_ptr)                                \
    {                                                                       \
        auto cout_buff = std::cout.rdbuf();                                 \
                                                                            \
        std::ostringstream local;                                           \
        std::cout.rdbuf(local.rdbuf());                                     \
        gko::write(std::cout, (*mat_st_ptr).mat);                           \
                                                                            \
        std::cout.rdbuf(cout_buff);                                         \
                                                                            \
        std::string str = local.str();                                      \
        char* cstr = new char[str.length() + 1];                            \
        std::strcpy(cstr, str.c_str());                                     \
        return cstr;                                                        \
    }

/**
 * @brief A build instruction for declaring gko::matrix::Dense<T> in the C API
 * header file
 */
#define DECLARE_DENSE_OVERLOAD(_ctype, _cpptype, _name)                      \
    struct gko_matrix_dense_##_name##_st;                                    \
    typedef struct gko_matrix_dense_##_name##_st* gko_matrix_dense_##_name;  \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create(           \
        gko_executor exec, gko_dim2_st size);                                \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_create_view(      \
        gko_executor exec, gko_dim2_st size, _ctype* values, size_t stride); \
    void ginkgo_matrix_dense_##_name##_delete(                               \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    void ginkgo_matrix_dense_##_name##_fill(                                 \
        gko_matrix_dense_##_name mat_st_ptr, _ctype value);                  \
    _ctype ginkgo_matrix_dense_##_name##_at(                                 \
        gko_matrix_dense_##_name mat_st_ptr, size_t row, size_t col);        \
    gko_dim2_st ginkgo_matrix_dense_##_name##_get_size(                      \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    _ctype* ginkgo_matrix_dense_##_name##_get_values(                        \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    const _ctype* ginkgo_matrix_dense_##_name##_get_const_values(            \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    size_t ginkgo_matrix_dense_##_name##_get_num_stored_elements(            \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    size_t ginkgo_matrix_dense_##_name##_get_stride(                         \
        gko_matrix_dense_##_name mat_st_ptr);                                \
    void ginkgo_matrix_dense_##_name##_compute_dot(                          \
        gko_matrix_dense_##_name mat_st_ptr1,                                \
        gko_matrix_dense_##_name mat_st_ptr2,                                \
        gko_matrix_dense_##_name mat_st_ptr_res);                            \
    void ginkgo_matrix_dense_##_name##_compute_norm1(                        \
        gko_matrix_dense_##_name mat_st_ptr1,                                \
        gko_matrix_dense_##_name mat_st_ptr2);                               \
    void ginkgo_matrix_dense_##_name##_compute_norm2(                        \
        gko_matrix_dense_##_name mat_st_ptr1,                                \
        gko_matrix_dense_##_name mat_st_ptr2);                               \
    gko_matrix_dense_##_name ginkgo_matrix_dense_##_name##_read(             \
        const char* str_ptr, gko_executor exec);                             \
    char* ginkgo_matrix_dense_##_name##_write_mtx(                           \
        gko_matrix_dense_##_name mat_st_ptr);


/**
 * @brief A build instruction for defining gko::matrix::Csr<Tv, Ti>, simplifying
 * its construction by removing the repetitive typing of array's name.
 *
 * @param _ctype_value  Type name of the element type in C
 * @param _ctype_index  Type name of the index type in C
 * @param _cpptype_value  Type name of the element type in C++
 * @param _cpptype_index  Type name of the index type in C++
 * @param _name  Name of the datatype of the sparse CSR matrix
 * @param _name_dense Name of the datatype of the dense matrix for apply
 * function
 */
#define DEFINE_CSR_OVERLOAD(_ctype_value, _ctype_index, _cpptype_value,        \
                            _cpptype_index, _name, _name_dense)                \
    struct gko_matrix_csr_##_name##_st {                                       \
        std::shared_ptr<gko::matrix::Csr<_cpptype_value, _cpptype_index>> mat; \
    };                                                                         \
                                                                               \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_create(                 \
        gko_executor exec, gko_dim2_st size, size_t nnz)                       \
    {                                                                          \
        return new gko_matrix_csr_##_name##_st{                                \
            gko::matrix::Csr<_cpptype_value, _cpptype_index>::create(          \
                (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols}, nnz)};  \
    }                                                                          \
                                                                               \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_create_view(            \
        gko_executor exec, gko_dim2_st size, size_t nnz,                       \
        _ctype_index* row_ptrs, _ctype_index* col_idxs, _ctype_value* values)  \
    {                                                                          \
        return new gko_matrix_csr_##_name##_st{                                \
            gko::matrix::Csr<_cpptype_value, _cpptype_index>::create(          \
                (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols},         \
                gko::array<_cpptype_value>::view((*exec).shared_ptr, nnz,      \
                                                 values),                      \
                gko::array<_cpptype_index>::view((*exec).shared_ptr, nnz,      \
                                                 col_idxs),                    \
                gko::array<_cpptype_index>::view((*exec).shared_ptr,           \
                                                 size.rows + 1, row_ptrs))};   \
    }                                                                          \
                                                                               \
    void ginkgo_matrix_csr_##_name##_delete(gko_matrix_csr_##_name mat_st_ptr) \
    {                                                                          \
        delete mat_st_ptr;                                                     \
    }                                                                          \
                                                                               \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_read(                   \
        const char* str_ptr, gko_executor exec)                                \
    {                                                                          \
        std::string filename(str_ptr);                                         \
        std::ifstream ifs(filename, std::ifstream::in);                        \
                                                                               \
        return new gko_matrix_csr_##_name##_st{                                \
            gko::read<gko::matrix::Csr<_cpptype_value, _cpptype_index>>(       \
                std::move(ifs), (*exec).shared_ptr)};                          \
    }                                                                          \
                                                                               \
    void ginkgo_write_csr_##_name##_in_coo(const char* str_ptr,                \
                                           gko_matrix_csr_##_name mat_st_ptr)  \
    {                                                                          \
        std::string filename(str_ptr);                                         \
        std::ofstream stream{filename};                                        \
        std::cerr << "Writing " << filename << std::endl;                      \
        gko::write(stream, (*mat_st_ptr).mat, gko::layout_type::coordinate);   \
    }                                                                          \
                                                                               \
    size_t ginkgo_matrix_csr_##_name##_get_num_stored_elements(                \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_num_stored_elements();                   \
    }                                                                          \
                                                                               \
    size_t ginkgo_matrix_csr_##_name##_get_num_srow_elements(                  \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_num_srow_elements();                     \
    }                                                                          \
                                                                               \
    gko_dim2_st ginkgo_matrix_csr_##_name##_get_size(                          \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        auto dim = (*mat_st_ptr).mat->get_size();                              \
        return gko_dim2_st{dim[0], dim[1]};                                    \
    }                                                                          \
                                                                               \
    const _ctype_value* ginkgo_matrix_csr_##_name##_get_const_values(          \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_const_values();                          \
    }                                                                          \
                                                                               \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_col_idxs(        \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_const_col_idxs();                        \
    }                                                                          \
                                                                               \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_row_ptrs(        \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_const_row_ptrs();                        \
    }                                                                          \
                                                                               \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_srow(            \
        gko_matrix_csr_##_name mat_st_ptr)                                     \
    {                                                                          \
        return (*mat_st_ptr).mat->get_const_srow();                            \
    }                                                                          \
                                                                               \
    void ginkgo_matrix_csr_##_name##_apply(                                    \
        gko_matrix_csr_##_name mat_st_ptr,                                     \
        gko_matrix_dense_##_name_dense alpha,                                  \
        gko_matrix_dense_##_name_dense x, gko_matrix_dense_##_name_dense beta, \
        gko_matrix_dense_##_name_dense y)                                      \
    {                                                                          \
        mat_st_ptr->mat->apply(alpha->mat, x->mat, beta->mat, y->mat);         \
    }


/**
 * @brief A build instruction for declaring gko::matrix::Csr<Tv,Ti> in the C API
 * header file
 */
#define DECLARE_CSR_OVERLOAD(_ctype_value, _ctype_index, _cpptype_value,       \
                             _cpptype_index, _name, _name_dense)               \
    struct gko_matrix_csr_##_name##_st;                                        \
    typedef struct gko_matrix_csr_##_name##_st* gko_matrix_csr_##_name;        \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_create(                 \
        gko_executor exec, gko_dim2_st size, size_t nnz);                      \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_create_view(            \
        gko_executor exec, gko_dim2_st size, size_t nnz,                       \
        _ctype_index* row_ptrs, _ctype_index* col_idxs, _ctype_value* values); \
    void ginkgo_matrix_csr_##_name##_delete(                                   \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    gko_matrix_csr_##_name ginkgo_matrix_csr_##_name##_read(                   \
        const char* str_ptr, gko_executor exec);                               \
    void ginkgo_write_csr_##_name##_in_coo(const char* str_ptr,                \
                                           gko_matrix_csr_##_name mat_st_ptr); \
    size_t ginkgo_matrix_csr_##_name##_get_num_stored_elements(                \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    size_t ginkgo_matrix_csr_##_name##_get_num_srow_elements(                  \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    gko_dim2_st ginkgo_matrix_csr_##_name##_get_size(                          \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    const _ctype_value* ginkgo_matrix_csr_##_name##_get_const_values(          \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_col_idxs(        \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_row_ptrs(        \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    const _ctype_index* ginkgo_matrix_csr_##_name##_get_const_srow(            \
        gko_matrix_csr_##_name mat_st_ptr);                                    \
    void ginkgo_matrix_csr_##_name##_apply(                                    \
        gko_matrix_csr_##_name mat_st_ptr,                                     \
        gko_matrix_dense_##_name_dense alpha,                                  \
        gko_matrix_dense_##_name_dense x, gko_matrix_dense_##_name_dense beta, \
        gko_matrix_dense_##_name_dense y);

#ifdef __cplusplus
extern "C" {
#endif
/* ----------------------------------------------------------------------
 * C memory management
 * ---------------------------------------------------------------------- */
void ginkgo_c_char_ptr_free(char* ptr);

/* ----------------------------------------------------------------------
 * Library functions for retrieving configuration information in GINKGO
 * ---------------------------------------------------------------------- */
/**
 * @brief This function is a wrapper for obtaining the version of the ginkgo
 * library
 *
 */
void ginkgo_version_get();

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
 * Library functions for executors (Creation, Getters) in GINKGO
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

/**
 * @brief Returns the master OmpExecutor of this Executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the current executor
 * @return gko_executor Raw pointer to the shared pointer of the master executor
 */
gko_executor ginkgo_executor_get_master(gko_executor exec_st_ptr);

/**
 * @brief Verifies whether the executors share the same memory.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the current executor
 * @param other_exec_st_ptr Raw pointer to the shared pointer of the other
 * executor
 */
bool ginkgo_executor_memory_accessible(gko_executor exec_st_ptr,
                                       gko_executor other_exec_st_ptr);

/**
 * @brief Synchronize the operations launched on the executor with its master.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the current executor
 */
void ginkgo_executor_synchronize(gko_executor exec_st_ptr);

//---------------------------- CPU -----------------------------
/**
 * @brief Create an OMP executor
 *
 * @return gko_executor Raw pointer to the shared pointer of the OMP executor
 * created
 */
gko_executor ginkgo_executor_omp_create();

/**
 * @brief Create a reference executor
 *
 * @return gko_executor Raw pointer to the shared pointer of the reference
 * executor created
 */
gko_executor ginkgo_executor_reference_create();

/**
 * @brief Get the number of cores of the CPU associated to this executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the executor
 * @return size_t No. of cores
 */
size_t ginkgo_executor_cpu_get_num_cores(gko_executor exec_st_ptr);

/**
 * @brief Get the number of threads per core of the CPU associated to this
 * executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the executor
 * @return size_t No. of threads per core
 */
size_t ginkgo_executor_cpu_get_num_threads_per_core(gko_executor exec_st_ptr);

//---------------------------- GPU -----------------------------
/**
 * @brief Get the device id of the device associated to this executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the executor
 * @return size_t Device id
 */
size_t ginkgo_executor_gpu_get_device_id(gko_executor exec_st_ptr);

// CUDA/HIP
/**
 * @brief Create a CUDA executor
 *
 * @param device_id Device id
 * @param exec_st_ptr Raw pointer to the shared pointer of the master executor
 * @return gko_executor Raw pointer to the shared pointer of the CUDA executor
 * created
 */
gko_executor ginkgo_executor_cuda_create(size_t device_id,
                                         gko_executor exec_st_ptr);

/**
 * @brief Get the number of devices of this CUDA executor.
 *
 * @return size_t No. of devices
 */
size_t ginkgo_executor_cuda_get_num_devices();

/**
 * @brief Create a HIP executor
 *
 * @param device_id Device id
 * @param exec_st_ptr Raw pointer to the shared pointer of the master executor
 * @return gko_executor Raw pointer to the shared pointer of the HIP executor
 * created
 */
gko_executor ginkgo_executor_hip_create(size_t device_id,
                                        gko_executor exec_st_ptr);

/**
 * @brief Get the number of devices of this HIP executor.
 *
 * @return size_t No. of devices
 */
size_t ginkgo_executor_hip_get_num_devices();

/**
 * @brief Get the number of multiprocessors of this thread-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t No. multiprocessors
 */
size_t ginkgo_executor_gpu_thread_get_num_multiprocessor(
    gko_executor exec_st_ptr);

/**
 * @brief Get the number of warps per SM of this thread-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t No. of warps per SM
 */
size_t ginkgo_executor_gpu_thread_get_num_warps_per_sm(
    gko_executor exec_st_ptr);

/**
 * @brief Get the number of warps of this thread-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t No. of warps
 */
size_t ginkgo_executor_gpu_thread_get_num_warps(gko_executor exec_st_ptr);

/**
 * @brief Get the warp size of this thread-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t The warp size of this executor
 */
size_t ginkgo_executor_gpu_thread_get_warp_size(gko_executor exec_st_ptr);

/**
 * @brief Get the major version of compute capability.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t The major version of compute capability
 */
size_t ginkgo_executor_gpu_thread_get_major_version(gko_executor exec_st_ptr);

/**
 * @brief Get the minor version of compute capability.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t The minor version of compute capability
 */
size_t ginkgo_executor_gpu_thread_get_minor_version(gko_executor exec_st_ptr);

/**
 * @brief Get the closest NUMA node.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the thread-based
 * executor
 * @return size_t No. of the closest NUMA node
 */
size_t ginkgo_executor_gpu_thread_get_closest_numa(gko_executor exec_st_ptr);

// DPCPP

/**
 * @brief Create a DPCPP executor
 *
 * @param device_id Device id
 * @param exec_st_ptr Raw pointer to the shared pointer of the master executor
 * @return gko_executor Raw pointer to the shared pointer of the DPCPP executor
 * created
 */
gko_executor ginkgo_executor_dpcpp_create(size_t device_id,
                                          gko_executor exec_st_ptr);

/**
 * @brief Get the number of devices of this DPCPP executor.
 *
 * @return size_t No. of devices
 */
size_t ginkgo_executor_dpcpp_get_num_devices();

/**
 * @brief Get the number of subgroups of this item-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the item-based
 * executor
 * @return size_t No. of subgroups
 */
size_t ginkgo_executor_gpu_item_get_max_subgroup_size(gko_executor exec_st_ptr);

/**
 * @brief Get the number of workgroups of this item-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the item-based
 * executor
 * @return size_t No. of workgroups
 */
size_t ginkgo_executor_gpu_item_get_max_workgroup_size(
    gko_executor exec_st_ptr);

/**
 * @brief Get the number of computing units of this item-based executor.
 *
 * @param exec_st_ptr Raw pointer to the shared pointer of the item-based
 * executor
 * @return size_t No. of computing units
 */
size_t ginkgo_executor_gpu_item_get_num_computing_units(
    gko_executor exec_st_ptr);

/* ----------------------------------------------------------------------
 * Library functions for creating arrays and array operations in GINKGO
 * ---------------------------------------------------------------------- */
DECLARE_ARRAY_OVERLOAD(int16_t, int16_t, i16)
DECLARE_ARRAY_OVERLOAD(int, int, i32)
DECLARE_ARRAY_OVERLOAD(int64_t, std::int64_t, i64)
DECLARE_ARRAY_OVERLOAD(float, float, f32)
DECLARE_ARRAY_OVERLOAD(double, double, f64)

/* ----------------------------------------------------------------------
 * Library functions for creating matrices and matrix operations in GINKGO
 * ---------------------------------------------------------------------- */
DECLARE_DENSE_OVERLOAD(float, float, f32)
DECLARE_DENSE_OVERLOAD(double, double, f64)

DECLARE_CSR_OVERLOAD(float, int, float, int, f32_i32, f32)
DECLARE_CSR_OVERLOAD(float, int64_t, float, std::int64_t, f32_i64, f32)
DECLARE_CSR_OVERLOAD(double, int, double, int, f64_i32, f64)
DECLARE_CSR_OVERLOAD(double, int64_t, double, std::int64_t, f64_i64, f64)

/* ----------------------------------------------------------------------
 * Library functions for deferred factory parameters in GINKGO
 * ---------------------------------------------------------------------- */
/**
 * @brief Struct containing the shared pointer to a ginkgo deferred factory
 * parameter
 *
 */
struct gko_deferred_factory_parameter_st;

/**
 * @brief Type of the pointer to the wrapped `gko_deferred_factory_parameter_st`
 * struct
 */
typedef struct gko_deferred_factory_parameter_st*
    gko_deferred_factory_parameter;

/**
 * @brief Deallocates memory for a ginkgo deferred factory parameter object.
 *
 * @param dfp_st_ptr Raw pointer to the shared pointer of the deferred factory
 * parameter object to be deleted
 */
void ginkgo_deferred_factory_parameter_delete(
    gko_deferred_factory_parameter dfp_st_ptr);

//-------------------- Preconditioner -----------------------------
/**
 * @brief Create a deferred factory parameter for an empty preconditioner
 *
 * @return gko_deferred_factory_parameter Raw pointer to the shared pointer of
 * the none preconditioner created
 */
gko_deferred_factory_parameter ginkgo_preconditioner_none_create();

gko_deferred_factory_parameter ginkgo_preconditioner_jacobi_f64_i32_create(
    int blocksize);
gko_deferred_factory_parameter ginkgo_preconditioner_ilu_f64_i32_create(
    gko_deferred_factory_parameter dfp_st_ptr);

//-------------------- Factorization ------------------------------
gko_deferred_factory_parameter ginkgo_factorization_parilu_f64_i32_create(
    int iteration, bool skip_sorting);

/* ----------------------------------------------------------------------
 * Library functions for LinOp objects in GINKGO
 * ---------------------------------------------------------------------- */
/**
 * @brief Struct containing the shared pointer to a ginkgo LinOp object
 *
 */
struct gko_linop_st;

/**
 * @brief Type of the pointer to the wrapped `gko_linop_st` struct
 *
 */
typedef struct gko_linop_st* gko_linop;

/**
 * @brief Deallocates memory for a ginkgo LinOp object.
 *
 * @param linop_st_ptr Raw pointer to the shared pointer of the LinOp object to
 * be deleted
 */
void ginkgo_linop_delete(gko_linop linop_st_ptr);

/**
 * @brief Applies a linear operator to a vector (or a sequence of vectors).
 *
 * @param A_st_ptr Raw pointer to the shared pointer of the LinOp object
 * @param b_st_ptr Raw pointer to the shared pointer of the input vector(s) on
 * which the operator is applied
 * @param x_st_ptr Raw pointer to the shared pointer of the output vector where
 * the result is stored
 */
void ginkgo_linop_apply(gko_linop A_st_ptr, gko_linop b_st_ptr,
                        gko_linop x_st_ptr);

//-------------------- Iterative solvers -----------------------------
gko_linop ginkgo_linop_cg_preconditioned_f64_create(
    gko_executor exec_st_ptr, gko_linop A_st_ptr,
    gko_deferred_factory_parameter dfp_st_ptr, double reduction, int maxiter);

gko_linop ginkgo_linop_gmres_preconditioned_f64_create(
    gko_executor exec_st_ptr, gko_linop A_st_ptr,
    gko_deferred_factory_parameter dfp_st_ptr, double reduction, int maxiter);

//-------------------- Direct solvers -----------------------------
gko_linop ginkgo_linop_spd_direct_f64_i64_create(gko_executor exec_st_ptr,
                                                 gko_linop A_st_ptr);

gko_linop ginkgo_linop_lu_direct_f64_i64_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr);

gko_linop ginkgo_linop_lu_direct_f64_i32_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr);

gko_linop ginkgo_linop_lu_direct_f32_i32_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr);

#ifdef __cplusplus
}
#endif /* language linkage */

#endif /* C_API_H */
