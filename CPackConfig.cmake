# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

include("Release/CPackConfig.cmake")
set(CPACK_INSTALL_CMAKE_PROJECTS
    "Debug;Ginkgo;All;/"
    "Release;Ginkgo;All;/"
    )

