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

void init_dense(py::module_& module_matrix)
{
    py::class_<gko::matrix::Dense<ValueType>,
               std::shared_ptr<gko::matrix::Dense<ValueType>>, gko::LinOp>(
        module_matrix, "Dense", py::buffer_protocol())
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::buffer b) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<ValueType>::format())
                throw std::runtime_error("Incompatible dtype");

            auto ref = gko::ReferenceExecutor::create();

            /* create a view into numpy data */
            auto elems = (info.ndim == 1) ? info.shape[0]
                                          : info.shape[0] * info.shape[1];

            auto view = gko::array<ValueType>(ref, elems, (ValueType*)info.ptr);

            auto rows = info.shape[0];
            auto cols = (info.ndim == 1) ? 1 : info.shape[1];

            // TODO fix dim<2>
            // TODO fix stride since the stride is given in bytes on the numpy
            // side
            return gko::matrix::Dense<ValueType>::create(
                exec, gko::dim<2>{rows, cols}, view, cols);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {
            return gko::matrix::Dense<ValueType>::create(exec);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim) {
            return gko::matrix::Dense<ValueType>::create(
                exec,
                gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()});
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                         size_t stride) {
            return gko::matrix::Dense<ValueType>::create(
                exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                stride);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                         gko::array<ValueType> view, size_t stride) {
            return gko::matrix::Dense<ValueType>::create(
                exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                view, stride);
        }))
        .def("__repr__",
             [](const gko::matrix::Dense<ValueType>& o) {
                 auto str = std::string("pygko.matrix.Dense object of size ");
                 auto elems = o.get_num_stored_elements();
                 str += std::to_string(elems);
                 if (o.get_executor() == o.get_executor()->get_master()) {
                     str += " on host";
                     if (elems < 10) {
                         str += " [ ";
                         for (int i = 0; i < elems; i++) {
                             str += std::to_string(o.at(i));
                             str += " ";
                         }
                     }
                     str += " ] ";
                 }
                 return str;
             })
        .def("copy_to_host",
             [](gko::matrix::Dense<ValueType>& m) {
                 auto host_exec = m.get_executor()->get_master();
                 std::cout << __FILE__ << "Warning creating a copy of dense\n";
                 if (m.get_executor() != host_exec) {
                     auto host_dense = gko::share(
                         gko::matrix::Dense<ValueType>::create(host_exec));
                     host_dense->operator=(m);
                     return host_dense;
                 } else {
                     return std::make_shared<gko::matrix::Dense<ValueType>>(m);
                 }
             })
        .def_buffer([](gko::matrix::Dense<ValueType>& m) -> py::buffer_info {
            // buffer info needs data on host, thus if data is on device it
            // should be copied to host first
            size_t rows = m.get_num_stored_elements() / m.get_stride();
            size_t cols = m.get_stride();
            size_t dim = (m.get_stride() == 1) ? 1 : 2;

            ValueType* buffer_ptr = nullptr;
            if (m.get_executor() != m.get_executor()->get_master()) {
                GKO_NOT_IMPLEMENTED;
            }
            buffer_ptr = m.get_values();

            if (dim == 1) {
                return py::buffer_info(
                    buffer_ptr,        /* Pointer to buffer */
                    sizeof(ValueType), /* Size of one scalar */
                    py::format_descriptor<ValueType>::format(), /* Python
                                                                struct-style
                                                                format
                                                                descriptor */
                    dim,                /* Number of dimensions */
                    {rows},             /* Buffer dimensions */
                    {sizeof(ValueType)} /* Strides (in bytes) for each index */
                );
            } else {
                return py::buffer_info(
                    buffer_ptr,        /* Pointer to buffer */
                    sizeof(ValueType), /* Size of one scalar */
                    py::format_descriptor<ValueType>::format(), /* Python
                                                                struct-style
                                                                format
                                                                descriptor */
                    dim,          /* Number of dimensions */
                    {rows, cols}, /* Buffer dimensions */
                    {sizeof(ValueType), sizeof(ValueType) * rows}
                    /* Strides (in bytes) for each index */
                );
            }
        })
        .def("get_stride", &gko::matrix::Dense<ValueType>::get_stride,
             "Returns the stride of the matrix.")
        // TODO checkout
        // https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
        .def("scale",
             [](gko::matrix::Dense<ValueType>& m, ValueType s) {
                 auto o = gko::matrix::Dense<ValueType>::create(
                     m.get_executor(), gko::dim<2>(1, 1));
                 o->fill(s);
                 m.scale(o);
             })
        .def("inv_scale", &gko::matrix::Dense<ValueType>::inv_scale,
             "Scales the matrix with the inverse of a scalar.")
        .def("add_scaled", &gko::matrix::Dense<ValueType>::add_scaled,
             "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
        .def(
            "sub_scaled", &gko::matrix::Dense<ValueType>::sub_scaled,
            "Subtracts `b` scaled by `alpha` fron the matrix (aka: BLAS axpy).")
        //       .def("compute_dot",
        //            py::overload_cast<const gko::LinOp*, gko::LinOp*>(
        //                &gko::matrix::Dense<ValueType>::compute_dot,
        //                py::const_),
        //            "Computes the column-wise dot product of this matrix and
        //            `b`")
        //        .def("compute_conj_dot",
        //             py::overload_cast<const gko::LinOp*, gko::LinOp*>(
        //                 &gko::matrix::Dense<ValueType>::compute_conj_dot,
        //                 py::const_),
        //             "Computes the column-wise dot product of `conj(this
        //             matrix)` and "
        //             "`b`.")
        //      .def("compute_norm2",
        //           py::overload_cast<gko::LinOp*>(
        //               &gko::matrix::Dense<ValueType>::compute_norm2,
        //               py::const_),
        //           "Computes the column-wise Euclidian (L^2) norm of this
        //           matrix.")
        //        .def("compute_norm1",
        //             py::overload_cast<gko::LinOp*>(
        //                 &gko::matrix::Dense<ValueType>::compute_norm1,
        //                 py::const_),
        //             "Computes the column-wise (L^1) norm of this matrix.")
        .def("at",
             static_cast<ValueType& (gko::matrix::Dense<ValueType>::*)(size_t)>(
                 &gko::matrix::Dense<ValueType>::at),
             "Returns an element using linearized index.")
        .def("at",
             static_cast<ValueType& (gko::matrix::Dense<ValueType>::*)(size_t,
                                                                       size_t)>(
                 &gko::matrix::Dense<ValueType>::at),
             "Returns an element at row, column index.")
        .def("get_num_stored_elements",
             &gko::matrix::Dense<ValueType>::get_num_stored_elements,
             "Returns the number of elements explicitly stored in the "
             "matrix.");
}
