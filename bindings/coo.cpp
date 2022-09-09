/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "python.hpp"

namespace py = pybind11;

// TODO wrap this for other data types like float, int ...

void init_coo(py::module_& module_matrix)
{
    py::class_<gko::matrix::Coo<ValueType>,
               std::shared_ptr<gko::matrix::Coo<ValueType>>, gko::LinOp>(
        module_matrix, "Coo", py::buffer_protocol())
        // .def(py::init([](std::shared_ptr<gko::Executor> exec, py::buffer b) {
        //     /* Request a buffer descriptor from Python */
        //     py::buffer_info info = b.request();
        //     auto ref = gko::ReferenceExecutor::create();

        //     /* create a view into numpy data */
        //     auto elems = (info.ndim == 1) ? info.shape[0]
        //                                   : info.shape[0] * info.shape[1];

        //     auto view = gko::array<ValueType>(ref, elems,
        //     (ValueType*)info.ptr);

        //     auto rows = info.shape[0];
        //     auto cols = (info.ndim == 1) ? 1 : info.shape[1];

        //     // TODO fix dim<2>
        //     // TODO fix stride since the stride is given in bytes on the
        //     numpy
        //     // side
        //     return gko::matrix::Dense<ValueType>::create(
        //         exec, gko::dim<2>{rows, cols}, view, cols);
        // }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {
            return gko::matrix::Coo<ValueType>::create(exec);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                         gko::array<ValueType>& vals,
                         gko::array<IndexType>& cols,
                         gko::array<IndexType>& rows) {
            return gko::share(gko::matrix::Coo<ValueType>::create(
                exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                vals, cols, rows));
        }))
        // .def(py::init([](std::shared_ptr<gko::Executor> exec, gko::dim<2>
        // dim,
        //                  size_t stride) {
        //     return gko::matrix::Dense<ValueType>::create(exec, dim, stride);
        // }))
        // .def(py::init([](std::shared_ptr<gko::Executor> exec, gko::dim<2>
        // dim,
        //                  gko::array<ValueType> view, size_t stride) {
        //     return gko::matrix::Dense<ValueType>::create(exec, dim, view,
        //                                                  stride);
        // }))
        .def("__repr__",
             [](const gko::matrix::Coo<ValueType>& o) {
                 auto str = std::string("pygko.matrix.Coo object");
                 return str;
             })
        // .def_buffer([](gko::matrix::Dense<ValueType>& m) -> py::buffer_info {
        //     size_t rows = m.get_num_stored_elements() / m.get_stride();
        //     size_t cols = m.get_stride();
        //     size_t dim = (m.get_stride() == 1) ? 1 : 2;

        //     // TODO implement for 2D matrix
        //     if (dim == 1) {
        //         return py::buffer_info(
        //             m.get_values(),    /* Pointer to buffer */
        //             sizeof(ValueType), /* Size of one scalar */
        //             py::format_descriptor<ValueType>::format(), /* Python
        //                                                         struct-style
        //                                                         format
        //                                                         descriptor */
        //             dim,                /* Number of dimensions */
        //             {rows},             /* Buffer dimensions */
        //             {sizeof(ValueType)} /* Strides (in bytes) for each index
        //             */
        //         );
        //     } else {
        //         return py::buffer_info(
        //             m.get_values(),    /* Pointer to buffer */
        //             sizeof(ValueType), /* Size of one scalar */
        //             py::format_descriptor<ValueType>::format(), /* Python
        //                                                         struct-style
        //                                                         format
        //                                                         descriptor */
        //             dim,          /* Number of dimensions */
        //             {rows, cols}, /* Buffer dimensions */
        //             {sizeof(ValueType), sizeof(ValueType) * rows}
        //             /* Strides (in bytes) for each index */
        //         );
        //     }
        // })
        // .def("get_stride", &gko::matrix::Dense<ValueType>::get_stride,
        //      "Returns the stride of the matrix.")
        // .def("scale", &gko::matrix::Dense<ValueType>::scale,
        //      "Scales the matrix with a scalar (aka: BLAS scal).")
        // .def("inv_scale", &gko::matrix::Dense<ValueType>::inv_scale,
        //      "Scales the matrix with the inverse of a scalar.")
        // .def("add_scaled", &gko::matrix::Dense<ValueType>::add_scaled,
        //      "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
        // .def(
        //     "sub_scaled", &gko::matrix::Dense<ValueType>::sub_scaled,
        //     "Subtracts `b` scaled by `alpha` fron the matrix (aka: BLAS
        //     axpy).")
        // .def("compute_dot",
        //      py::overload_cast<const gko::LinOp*, gko::LinOp*>(
        //          &gko::matrix::Dense<ValueType>::compute_dot, py::const_),
        //      "Computes the column-wise dot product of this matrix and `b`")
        // .def("compute_conj_dot",
        //      py::overload_cast<const gko::LinOp*, gko::LinOp*>(
        //          &gko::matrix::Dense<ValueType>::compute_conj_dot,
        //          py::const_),
        //      "Computes the column-wise dot product of `conj(this matrix)` and
        //      "
        //      "`b`.")
        // .def("compute_norm2",
        //      py::overload_cast<gko::LinOp*>(
        //          &gko::matrix::Dense<ValueType>::compute_norm2, py::const_),
        //      "Computes the column-wise Euclidian (L^2) norm of this matrix.")
        // .def("compute_norm1",
        //      py::overload_cast<gko::LinOp*>(
        //          &gko::matrix::Dense<ValueType>::compute_norm1, py::const_),
        //      "Computes the column-wise (L^1) norm of this matrix.")
        // .def("at",
        //      static_cast<ValueType&
        //      (gko::matrix::Dense<ValueType>::*)(size_t)>(
        //          &gko::matrix::Dense<ValueType>::at),
        //      "Returns an element using linearized index.")
        // .def("at",
        //      static_cast<ValueType&
        //      (gko::matrix::Dense<ValueType>::*)(size_t,
        //                                                                size_t)>(
        //          &gko::matrix::Dense<ValueType>::at),
        //      "Returns an element at row, column index.")
        // .def("apply", &gko::matrix::Coo<ValueType>::apply, "");
        .def("get_num_stored_elements",
             &gko::matrix::Coo<ValueType>::get_num_stored_elements,
             "Applies Coo matrix axpy to a vector (or a sequence of vectors)."
             "Performs the operation x = Coo * b + x");
}
