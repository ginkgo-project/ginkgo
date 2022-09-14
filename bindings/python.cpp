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

void init_dense(py::module_&);
void init_coo(py::module_&);
void init_cg(py::module_&);
void init_stop(py::module_&);
void init_preconditioner(py::module_&);


PYBIND11_MODULE(pygko, m)
{
    m.doc() = "Python bindings for the Ginkgo framework";
    py::class_<gko::Executor, std::shared_ptr<gko::Executor>> Executor(
        m, "Executor");

    py::class_<gko::OmpExecutor, std::shared_ptr<gko::OmpExecutor>>(
        m, "OmpExecutor", Executor)
        .def(py::init(&gko::OmpExecutor::create));

    py::class_<gko::ReferenceExecutor, std::shared_ptr<gko::ReferenceExecutor>>(
        m, "ReferenceExecutor", Executor)
        .def(py::init(&gko::ReferenceExecutor::create));

    py::enum_<gko::allocation_mode>(m, "allocation_mode")
        .value("device", gko::allocation_mode::device,
               "Allocates memory on the device and Unified Memory model is not "
               "used.")
        .value("unified_global", gko::allocation_mode::unified_global,
               "Allocates memory on the device, but is accessible by the host "
               "through the Unified memory model.")
        .value("unified_host", gko::allocation_mode::unified_host,
               "Allocates memory on the host and it is not available on "
               "devices which do not have concurrent acesses switched on, but "
               "this access can be explictly switched on, "
               "when necessary.")
        .export_values();

    py::class_<gko::CudaExecutor, std::shared_ptr<gko::CudaExecutor>>(
        m, "CudaExecutor", Executor)
        .def(py::init(&gko::CudaExecutor::create), py::arg("device_id") = 0,
             py::arg("master") = gko::OmpExecutor::create(),
             py::arg("device_reset") = false,
             py::arg("alloc_mode") = gko::allocation_mode::unified_global);

    py::class_<gko::HipExecutor, std::shared_ptr<gko::HipExecutor>>(
        m, "HipExecutor", Executor)
        .def(py::init(&gko::HipExecutor::create));

    py::class_<gko::DpcppExecutor, std::shared_ptr<gko::DpcppExecutor>>(
        m, "DpcppExecutor", Executor)
        .def(py::init(&gko::DpcppExecutor::create));

    py::class_<gko::array<double>>(m, "array")
        .def(py::init<std::shared_ptr<const gko::Executor>, int>())
        .def("fill", &gko::array<double>::fill,
             "Fill the array with the given value.")
        .def("get_num_elems", &gko::array<double>::get_num_elems);

    py::class_<gko::dim<2>>(m, "dim2").def(py::init<int, int>());

    py::class_<gko::LinOp, std::shared_ptr<gko::LinOp>> LinOp(m, "LinOp");

    py::module_ module_matrix = m.def_submodule(
        "matrix", "Submodule for Ginkgos matrix format bindings");

    py::module_ module_solver =
        m.def_submodule("solver", "Submodule for Ginkgos solver bindings");

    py::module_ module_preconditioner = m.def_submodule(
        "preconditioner", "Submodule for Ginkgos preconditioner bindings");

    py::module_ module_stop = m.def_submodule(
        "stop", "Submodule for Ginkgos stopping criteria bindings");

    m.def("read_dense",
          [](const std::string& fn, std::shared_ptr<gko::Executor> exec) {
              return gko::read<gko::matrix::Dense<ValueType>>(std::ifstream(fn),
                                                              exec);
          });

    m.def("read_coo",
          [](const std::string& fn, std::shared_ptr<gko::Executor> exec) {
              return gko::share(gko::read<gko::matrix::Coo<ValueType>>(
                  std::ifstream(fn), exec));
          });

    init_dense(module_matrix);
    init_coo(module_matrix);
    init_cg(module_solver);
    init_preconditioner(module_preconditioner);
    init_stop(module_stop);
}
