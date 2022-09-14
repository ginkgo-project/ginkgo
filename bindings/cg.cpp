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

void init_cg(py::module_& module_solver)
{
    py::class_<gko::solver::Cg<ValueType>,
               std::shared_ptr<gko::solver::Cg<ValueType>>, gko::LinOp>(
        module_solver, "Cg")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<gko::LinOp> system_matrix,
                         py::list with) {
            auto factory = gko::solver::Cg<ValueType>::build();

            std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
                stopping_criteria{};
            for (auto& w : with) {
                try {
                    stopping_criteria.push_back(
                        w.cast<std::shared_ptr<
                            const gko::stop::CriterionFactory>>());
                    continue;
                } catch (...) {
                }
                try {
                    factory.with_preconditioner(
                        w.cast<std::shared_ptr<const gko::LinOpFactory>>());
                    continue;
                } catch (...) {
                }
            }
            factory.with_criteria(stopping_criteria);

            auto factory_ = factory.on(exec);

            for (auto& l : with) {
                try {
                    factory_->add_logger(
                        l.cast<std::shared_ptr<const gko::log::Logger>>());
                } catch (...) {
                }
            }

            return factory_->generate(system_matrix);
        }))
        .def("apply",
             py::overload_cast<const gko::LinOp*, gko::LinOp*>(
                 &gko::solver::Cg<ValueType>::apply),
             "")
        .def("apply",
             py::overload_cast<const gko::LinOp*, gko::LinOp*>(
                 &gko::solver::Cg<ValueType>::apply, py::const_),
             "");
}
