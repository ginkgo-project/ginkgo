// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#ifndef PYTHON_CSL_INNER_FUNCTIONS
#define PYTHON_CSL_INNER_FUNCTIONS                                             \
    ""                                                                         \
    "\n"                                                                       \
    "import os"                                                                \
    "\n"                                                                       \
    "import numpy as np"                                                       \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, "    \
    "MemcpyOrder"                                                              \
    "\n"                                                                       \
    "from cerebras.sdk.client import SdkRuntime"                               \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "runner = None"                                                            \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "def start_device("                                                        \
    "\n"                                                                       \
    "        simulator: bool"                                                  \
    "\n"                                                                       \
    "    ) -> None:"                                                           \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    "    This function loads the code onto the accelerator (device)"           \
    "\n"                                                                       \
    "    and prepares the runtime environment. Call only once"                 \
    "\n"                                                                       \
    "    at the start of whatever you are doing."                              \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    @param: simulator           Should a simulator be used (True) or "    \
    "real"                                                                     \
    "\n"                                                                       \
    "                                hardware (False)?"                        \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # assign to global runner variable"                                   \
    "\n"                                                                       \
    "    global runner"                                                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # check for wrong call sequence"                                      \
    "\n"                                                                       \
    "    if runner is not None:"                                               \
    "\n"                                                                       \
    "        raise RuntimeError('start_device can only be called once')"       \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # get artifact path"                                                  \
    "\n"                                                                       \
    "    csl_path = '" EXECDIR "' + '/' + '" CSLFILENAME                       \
    "'"                                                                        \
    "\n"                                                                       \
    "    if not os.path.exists(csl_path):"                                     \
    "\n"                                                                       \
    "        raise RuntimeError("                                              \
    "\n"                                                                       \
    "            '''The csl executable could not be found."                    \
    "\n"                                                                       \
    "                Please compile the code first'''"                         \
    "\n"                                                                       \
    "        )"                                                                \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # construct runtime"                                                  \
    "\n"                                                                       \
    "    runner = SdkRuntime(csl_path, simulator=simulator)"                   \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # load code onto device"                                              \
    "\n"                                                                       \
    "    runner.__enter__()"                                                   \
    "\n"                                                                       \
    "    print('INIT CS2')"                                                    \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "def stop_device() -> None:"                                               \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    "    This function unloads the code from the accelerator (device)"         \
    "\n"                                                                       \
    "    and shutdowns the runtime environment. Call only once"                \
    "\n"                                                                       \
    "    at the end of whatever you are doing."                                \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # modify global runner variable"                                      \
    "\n"                                                                       \
    "    global runner"                                                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # check for error"                                                    \
    "\n"                                                                       \
    "    if runner is None:"                                                   \
    "\n"                                                                       \
    "        raise RuntimeError('stop_device must be called after "            \
    "start_device')"                                                           \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # try to stop device"                                                 \
    "\n"                                                                       \
    "    runner.__exit__(None, None, None)"                                    \
    "\n"                                                                       \
    "    print('EXIT CS2')"                                                    \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "def copy_h2d_f32("                                                        \
    "\n"                                                                       \
    "        target_var: str,"                                                 \
    "\n"                                                                       \
    "        matrix: np.ndarray,"                                              \
    "\n"                                                                       \
    "        offset_1: int,"                                                   \
    "\n"                                                                       \
    "        offset_2: int,"                                                   \
    "\n"                                                                       \
    "        size_1: int,"                                                     \
    "\n"                                                                       \
    "        size_2: int,"                                                     \
    "\n"                                                                       \
    "        elements_per_pe: int,"                                            \
    "\n"                                                                       \
    "        streaming: bool,"                                                 \
    "\n"                                                                       \
    "        nonblock: bool,"                                                  \
    "\n"                                                                       \
    "    ) -> None:"                                                           \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    "    This function copies an array of length size1 x size2 onto the"       \
    "\n"                                                                       \
    "    accelerator (device)."                                                \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    @param: target_var          Name of the exported symbol in the csl"   \
    "\n"                                                                       \
    "                                to copy the array to."                    \
    "\n"                                                                       \
    "    @param: matrix              The array to copy to the device of size"  \
    "\n"                                                                       \
    "                                size1 x size2 x 4byte (single "           \
    "precision)."                                                              \
    "\n"                                                                       \
    "    @param: offset_1            The offset in the first direction to"     \
    "\n"                                                                       \
    "                                place the start of the matrix onto."      \
    "\n"                                                                       \
    "    @param: offset_2            The offset in the second direction to"    \
    "\n"                                                                       \
    "                                place the start of the matrix onto."      \
    "\n"                                                                       \
    "    @param: size_1              The size of the matrix in the first"      \
    "\n"                                                                       \
    "                                direction (number of rows)."              \
    "\n"                                                                       \
    "    @param: size_2              The size of the matrix in the second"     \
    "\n"                                                                       \
    "                                direction (number of columns)."           \
    "\n"                                                                       \
    "    @param: elements_per_pe     How many elements should be placed on"    \
    "\n"                                                                       \
    "                                each PE/core? This number is product"     \
    "\n"                                                                       \
    "                                of the number of local rows x number"     \
    "\n"                                                                       \
    "                                of local columns, if you think of matrix" \
    "\n"                                                                       \
    "                                as a block matrix and each block"         \
    "\n"                                                                       \
    "                                belonging to a PE."                       \
    "\n"                                                                       \
    "    @param: streaming           Streaming operation mode. SET TO FALSE,"  \
    "\n"                                                                       \
    "                                WE DON'T KNOW WHAT TRUE DOES RIGHT NOW!"  \
    "\n"                                                                       \
    "    @param: nonblock            Should the copy be executed as blocking"  \
    "\n"                                                                       \
    "                                or nonblocking operation? CURRENTLY WE"   \
    "\n"                                                                       \
    "                                ALWAYS SET TO FALSE (blocking)."          \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # using the global runner"                                            \
    "\n"                                                                       \
    "    global runner"                                                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # check if there is some error"                                       \
    "\n"                                                                       \
    "    if runner is None:"                                                   \
    "\n"                                                                       \
    "        raise RuntimeError('start_device must be called before any "      \
    "other device API')"                                                       \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # Get symbols"                                                        \
    "\n"                                                                       \
    "    var = runner.get_id(target_var)"                                      \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # copy to device"                                                     \
    "\n"                                                                       \
    "    runner.memcpy_h2d("                                                   \
    "\n"                                                                       \
    "        var,"                                                             \
    "\n"                                                                       \
    "        matrix,"                                                          \
    "\n"                                                                       \
    "        offset_1,"                                                        \
    "\n"                                                                       \
    "        offset_2,"                                                        \
    "\n"                                                                       \
    "        size_1,"                                                          \
    "\n"                                                                       \
    "        size_2,"                                                          \
    "\n"                                                                       \
    "        elements_per_pe,"                                                 \
    "\n"                                                                       \
    "        streaming=streaming,"                                             \
    "\n"                                                                       \
    "        data_type=MemcpyDataType.MEMCPY_32BIT,"                           \
    "\n"                                                                       \
    "        order=MemcpyOrder.ROW_MAJOR,"                                     \
    "\n"                                                                       \
    "        nonblock=nonblock"                                                \
    "\n"                                                                       \
    "    )"                                                                    \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "def copy_d2h_f32("                                                        \
    "\n"                                                                       \
    "        target_var: str,"                                                 \
    "\n"                                                                       \
    "        matrix: np.ndarray,"                                              \
    "\n"                                                                       \
    "        offset_1: int,"                                                   \
    "\n"                                                                       \
    "        offset_2: int,"                                                   \
    "\n"                                                                       \
    "        size_1: int,"                                                     \
    "\n"                                                                       \
    "        size_2: int,"                                                     \
    "\n"                                                                       \
    "        elements_per_pe: int,"                                            \
    "\n"                                                                       \
    "        streaming: bool,"                                                 \
    "\n"                                                                       \
    "        nonblock: bool,"                                                  \
    "\n"                                                                       \
    "    ) -> None:"                                                           \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    "    This function copies an array of length size1 x size2 from the"       \
    "\n"                                                                       \
    "    accelerator (device) to the host."                                    \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    @param: target_var          Name of the exported symbol in the csl"   \
    "\n"                                                                       \
    "                                to copy the array from."                  \
    "\n"                                                                       \
    "    @param: matrix              The array to store the copy into of size" \
    "\n"                                                                       \
    "                                size1 x size2 x 4byte (single "           \
    "precision)."                                                              \
    "\n"                                                                       \
    "    @param: offset_1            The offset in the first direction to"     \
    "\n"                                                                       \
    "                                place the start of the matrix onto."      \
    "\n"                                                                       \
    "    @param: offset_2            The offset in the second direction to"    \
    "\n"                                                                       \
    "                                place the start of the matrix onto."      \
    "\n"                                                                       \
    "    @param: size_1              The size of the matrix in the first"      \
    "\n"                                                                       \
    "                                direction (number of rows)."              \
    "\n"                                                                       \
    "    @param: size_2              The size of the matrix in the second"     \
    "\n"                                                                       \
    "                                direction (number of columns)."           \
    "\n"                                                                       \
    "    @param: elements_per_pe     How many elements are placed on"          \
    "\n"                                                                       \
    "                                each PE/core? This number is product"     \
    "\n"                                                                       \
    "                                of the number of local rows x number"     \
    "\n"                                                                       \
    "                                of local columns, if you think of matrix" \
    "\n"                                                                       \
    "                                as a block matrix and each block"         \
    "\n"                                                                       \
    "                                belonging to a PE."                       \
    "\n"                                                                       \
    "    @param: streaming           Streaming operation mode. SET TO FALSE,"  \
    "\n"                                                                       \
    "                                WE DON'T KNOW WHAT TRUE DOES RIGHT NOW!"  \
    "\n"                                                                       \
    "    @param: nonblock            Should the copy be executed as blocking"  \
    "\n"                                                                       \
    "                                or nonblocking operation? CURRENTLY WE"   \
    "\n"                                                                       \
    "                                ALWAYS SET TO FALSE (blocking)."          \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # using the global runner"                                            \
    "\n"                                                                       \
    "    global runner"                                                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # check if there is some error"                                       \
    "\n"                                                                       \
    "    if runner is None:"                                                   \
    "\n"                                                                       \
    "        raise RuntimeError('start_device must be called before any "      \
    "other device API')"                                                       \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # get symbol"                                                         \
    "\n"                                                                       \
    "    var = runner.get_id(target_var)"                                      \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # copy to device"                                                     \
    "\n"                                                                       \
    "    runner.memcpy_d2h("                                                   \
    "\n"                                                                       \
    "        matrix,"                                                          \
    "\n"                                                                       \
    "        var,"                                                             \
    "\n"                                                                       \
    "        offset_1,"                                                        \
    "\n"                                                                       \
    "        offset_2,"                                                        \
    "\n"                                                                       \
    "        size_1,"                                                          \
    "\n"                                                                       \
    "        size_2,"                                                          \
    "\n"                                                                       \
    "        elements_per_pe,"                                                 \
    "\n"                                                                       \
    "        streaming=streaming,"                                             \
    "\n"                                                                       \
    "        data_type=MemcpyDataType.MEMCPY_32BIT,"                           \
    "\n"                                                                       \
    "        order=MemcpyOrder.ROW_MAJOR,"                                     \
    "\n"                                                                       \
    "        nonblock=nonblock"                                                \
    "\n"                                                                       \
    "    )"                                                                    \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "def call_cerebras_func(target_func: str, nonblock: bool) -> None:"        \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    "    This function triggers the execution of a csl function on the"        \
    "\n"                                                                       \
    "    accelerator (device)."                                                \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    @param: target_func         Name of the exported function in the csl" \
    "\n"                                                                       \
    "                                to call."                                 \
    "\n"                                                                       \
    "    @param: nonblock            Should the function be executed as "      \
    "blocking"                                                                 \
    "\n"                                                                       \
    "                                or nonblocking operation? CURRENTLY WE"   \
    "\n"                                                                       \
    "                                ALWAYS SET TO FALSE (blocking)."          \
    "\n"                                                                       \
    "    '''"                                                                  \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # using the global runner"                                            \
    "\n"                                                                       \
    "    global runner"                                                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # check if there is some error"                                       \
    "\n"                                                                       \
    "    if runner is None:"                                                   \
    "\n"                                                                       \
    "        raise RuntimeError('start_device must be called before any "      \
    "other device API')"                                                       \
    "\n"                                                                       \
    ""                                                                         \
    "\n"                                                                       \
    "    # call the cerebras function"                                         \
    "\n"                                                                       \
    "    runner.launch(target_func, nonblock=nonblock)"                        \
    "\n"                                                                       \
    ""                                                                         \
    "\n"

#endif
