# License of Ginkgo

Ginkgo follows the [REUSE specification of the Free Software Foundation Europe](https://reuse.software/).
Therefore, you will find clear licensing information for each and every file in this repository.
This can either be an SPDX tag in the beginning of the file or a `*.license` file next to the file.

**The source code of Ginkgo is released under the terms of the BSD 3-Clause "New" or "Revised" License**.
However, the licensing terms for assets, input files, and included third-party software may differ.
All licenses used by this project are stored in the `LICENSES` directory.

If you want to use Ginkgo, but need a different license, please
[contact us](https://github.com/ginkgo-project/ginkgo/wiki#contact-us).

## Third-Party Licenses

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

> Copyright 2008, Google Inc. All rights reserved.
>
> SPDX-License-Identifier: BSD-3-Clause

When compiling Ginkgo with `-DGINKGO_DEVEL_TOOLS=ON` the build system will download
[git-cmake-format](https://github.com/kbenzie/git-cmake-format), available under
the following license:

> SPDX-License-Identifier: Unlicense

When compiling Ginkgo with `-DGINKGO_BUILD_BENCHMARKS=ON` the build system will
download, build, and link [gflags](https://github.com/gflags/gflags) and
[RapidJSON](https://github.com/Tencent/rapidjson) with the
benchmark suites. gtest is available under the following license:

> Copyright (c) 2006, Google Inc. All rights reserved.
>
> SPDX-License-Identifier: BSD-3-Clause

RapidJSON is available under the following license (note that Ginkgo's build
system automatically removes the `bin/jsonchecker/` directory which is licensed
under the problematic JSON license):

> Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip. All rights reserved.
>
> SPDX-License-Identifier: MIT
>
> Other dependencies and licenses:
> ---------------------------------
>
> The msinttypes r29
>
> Copyright (c) 2006-2013 Alexander Chemeris
>
> SPDX-License-Identifier: BSD-3-Clause


For generating the documentation of Ginkgo, some scripts from the deal.II
library are used. You can refer to the `doc/` folder to see which files are a
modified version of deal.II's documentation generation scripts. Additionally,
an example of interfacing Ginkgo with an external library is available in the
`examples/` folder of Ginkgo. This uses the `step-9` of deal.II which is
also licensed the same as the deal.II library.

> Copyright (C) 2000 - 2019 by the deal.II authors
>
> SPDX-License-Identifier: LGPL-2.1-or-later


__NOTE:__ Some of the options that pull additional software when compiling
Ginkgo are ON by default, and have to be disabled manually to prevent
third-party licensing. Refer to the [Installation section in
INSTALL.md](INSTALL.md#Building) for more details.
