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

When compiling Ginkgo with `-DGINKGO_DEVEL_TOOLS=ON` the build system will download
[git-cmake-format](https://github.com/kbenzie/git-cmake-format), available under
the following license:

> This is free and unencumbered software released into the public domain.
>
> Anyone is free to copy, modify, publish, use, compile, sell, or distribute
> this software, either in source code form or as a compiled binary, for any
> purpose, commercial or non-commercial, and by any means.
>
> In jurisdictions that recognize copyright laws, the author or authors of this
> software dedicate any and all copyright interest in the software to the public
> domain.  We make this dedication for the benefit of the public at large and to
> the detriment of our heirs and successors.  We intend this dedication to be an
> overt act of relinquishment in perpetuity of all present and future rights to
> this software under copyright law.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTBILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT, IN NO EVENT SHALL THE
> AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
> ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
> WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
>
> For more information, please refer to <http://unlicense.org/>

When compiling Ginkgo with `-DGINKGO_BUILD_BENCHMARKS=ON` the build system will
download, build, and link [gflags](https://github.com/gflags/gflags) and
[RapidJSON](https://github.com/Tencent/rapidjson) with the
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

RapidJSON is available under the following license (note that Ginkgo's build
system automatically removes the `bin/jsonchecker/` directory which is licensed
under the problematic JSON license):

> Tencent is pleased to support the open source community by making RapidJSON
> available.
>
> Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip.  All
> rights reserved.
>
> If you have downloaded a copy of the RapidJSON binary from Tencent, please
> note that the RapidJSON binary is licensed under the MIT License.  If you have
> downloaded a copy of the RapidJSON source code from Tencent, please note that
> RapidJSON source code is licensed under the MIT License, except for the
> third-party components listed below which are subject to different license
> terms.  Your integration of RapidJSON into your own projects may require
> compliance with the MIT License, as well as the other licenses applicable to
> the third-party components included within RapidJSON. To avoid the problematic
> JSON license in your own projects, it's sufficient to exclude the
> bin/jsonchecker/ directory, as it's the only code under the JSON license.  A
> copy of the MIT License is included in this file.
>
> Other dependencies and licenses:
>
> Open Source Software Licensed Under the BSD License:
> --------------------------------------------------------------------
>
> The msinttypes r29
>
> Copyright (c) 2006-2013 Alexander Chemeris
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of  copyright holder nor the names of its contributors may
>   be used to endorse or promote products derived from this software without
>   specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
> EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
> DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
> ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> Open Source Software Licensed Under the JSON License:
> --------------------------------------------------------------------
>
> json.org
> Copyright (c) 2002
> JSON.org All Rights Reserved.
>
> JSON_checker
> Copyright (c) 2002 JSON.org
> All Rights Reserved.
>
>
> Terms of the JSON License:
> ---------------------------------------------------
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> The Software shall be used for Good, not Evil.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
>
> Terms of the MIT License:
> --------------------------------------------------------------------
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
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

__NOTE:__ Some of the options that pull additional software when compiling
Ginkgo are ON by default, and have to be disabled manually to prevent
third-party licensing. Refer to the [Installation section in
INSTALL.md](INSTALL.md#Building) for more details.
