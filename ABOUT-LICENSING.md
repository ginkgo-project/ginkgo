About Licensing
===============

Ginkgo is available under the [3-clause BSD license](LICENSE).

If you want to use Ginkgo, but need a different license, please
[contact us](https://github.com/ginkgo-project/ginkgo/wiki#contact-us).

Ginkgo source distribution does not bundle any third-party software. However,
the licensing conditions of Ginkgo's dependencies still apply, depending on the
modules used with Ginkgo. Please refer to the documentation of the dependencies
for license conditions. Additionally, depending on the options used to compile
Ginkgo, the build system will pull additional dependencies which have their own
licensing conditions (note that all of these are extra utilities, and it is
possible to obtain a fully functional installation of Ginkgo without any of
them). They include the following:

When compiling Ginkgo with `-DGINKGO_BUILD_TEST=ON`, the build system will
download and build [Google Test](https://github.com/google/googletest), and link
the binary with Ginkgo's unit tests. Google Test is available under the
following license:

> Copyright 2008, Google Inc.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of Google Inc. nor the names of its contributors may be
>   used to endorse or promote products derived from this software without
>   specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

When compiling Ginkgo with `-DGINKGO_BUILD_BENCHMARKS=ON` the build system will
download, build, and link [gflags](https://github.com/gflags/gflags) and
[nlohmann-json](https://github.com/nlohmann/json) with the
benchmark suites. gtest is available under the following license:

> Copyright (c) 2006, Google Inc.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
> * Redistributions of source code must retain the above copyright
> notice, this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above
> copyright notice, this list of conditions and the following disclaimer
> in the documentation and/or other materials provided with the
> distribution.
> * Neither the name of Google Inc. nor the names of its
> contributors may be used to endorse or promote products derived from
> this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

nlohmann-json is available under the following license:

> MIT License 
> 
> Copyright (c) 2013-2022 Niels Lohmann
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

For generating the documentation of Ginkgo, some scripts from the deal.II
library are used. You can refer to the `doc/` folder to see which files are a
modified version of deal.II's documentation generation scripts. Additionally,
an example of interfacing Ginkgo with an external library is available in the
`examples/` folder of Ginkgo. This uses the `step-9` of deal.II which is
also licensed the same as the deal.II library.

> Copyright (C) 2000 - 2019 by the deal.II authors
>
> The deal.II library is free software; you can use it, redistribute it, and/or
> modify it under the terms of the GNU Lesser General Public License as published
> by the Free Software Foundation; either version 2.1 of the License, or (at your
> option) any later version.
> 
> The full text of the GNU Lesser General Public
> version 2.1 is available on the [deal.II repository
> page](https://github.com/dealii/dealii/blob/master/LICENSE.md) or on the
> official [GNU license page](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html).


The AMD reordering class inside Ginkgo wraps the SuiteSparse AMD reordering, which is available
under the following license:

>    AMD, Copyright (c), 1996-2022, Timothy A. Davis,
>    Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.
>
>    Availability:
>
>        http://suitesparse.com
>
>    -------------------------------------------------------------------------------
>    AMD License: BSD 3-clause:
>    -------------------------------------------------------------------------------
>
>        Redistribution and use in source and binary forms, with or without
>        modification, are permitted provided that the following conditions are met:
>            * Redistributions of source code must retain the above copyright
>              notice, this list of conditions and the following disclaimer.
>            * Redistributions in binary form must reproduce the above copyright
>              notice, this list of conditions and the following disclaimer in the
>              documentation and/or other materials provided with the distribution.
>            * Neither the name of the organizations to which the authors are
>              affiliated, nor the names of its contributors may be used to endorse
>              or promote products derived from this software without specific prior
>              written permission.
>
>        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
>        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
>        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
>        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
>        DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
>        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
>        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
>        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
>        LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
>        OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
>        DAMAGE.


For detecting the HWLOC library, we used a modified version of the FindHWLOC.cmake file from the MORSE-cmake library. The library is [available on gitlab](https://gitlab.inria.fr/solverstack/morse_cmake), and its LICENSE is available below:

> @copyright (c) 2012-2020 Inria. All rights reserved.
> @copyright (c) 2012-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
>
> This software is a computer program whose purpose is to process
> Matrices Over Runtime Systems @ Exascale (MORSE). More information
> can be found on the following website: http://www.inria.fr/en/teams/morse.
>
> This software is governed by the CeCILL-C license under French law and
> abiding by the rules of distribution of free software.  You can  use,
> modify and/ or redistribute the software under the terms of the CeCILL-C
> license as circulated by CEA, CNRS and INRIA at the following URL
> "http://www.cecill.info".
>
> As a counterpart to the access to the source code and  rights to copy,
> modify and redistribute granted by the license, users are provided only
> with a limited warranty  and the software's author,  the holder of the
> economic rights,  and the successive licensors  have only  limited
> liability.
>
> In this respect, the user's attention is drawn to the risks associated
> with loading,  using,  modifying and/or developing or reproducing the
> software by the user in light of its specific status of free software,
> that may mean  that it is complicated to manipulate,  and  that  also
> therefore means  that it is reserved for developers  and  experienced
> professionals having in-depth computer knowledge. Users are therefore
> encouraged to load and test the software's suitability as regards their
> requirements in conditions enabling the security of their systems and/or
> data to be ensured and,  more generally, to use and operate it in the
> same conditions as regards security.
>
> The fact that you are presently reading this means that you have had
> knowledge of the CeCILL-C license and that you accept its terms.



__NOTE:__ Some of the options that pull additional software when compiling
Ginkgo are ON by default, and have to be disabled manually to prevent
third-party licensing. Refer to the [Installation section in
INSTALL.md](INSTALL.md#Building) for more details.


When using testing with MPI switched on, the gtest-mpi-listener header only library is used for testing MPI functionality. The repository is licensed triple licensed under BSD-3, MIT and Apache 2.0. The License duplicated below. More details on the License and the library are [available on github](https://github.com/LLNL/gtest-mpi-listener)


> Copyright 2005, Google Inc.  All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
> 
> * Redistributions of source code must retain the above copyright
> notice, this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above
> copyright notice, this list of conditions and the following disclaimer
> in the documentation and/or other materials provided with the
> distribution.
> * Neither the name of Google Inc. nor the names of its
> contributors may be used to endorse or promote products derived from
> this software without specific prior written permission.
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
