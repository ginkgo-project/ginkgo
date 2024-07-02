#!/usr/bin/env python3
import sys
import os
import difflib
import subprocess

common_filename = sys.argv[1]
base_filename = common_filename.replace("common/cuda_hip/", "").replace(".hpp.inc", "")
cuda_filename = next(
    f"cuda/{base_filename}{extension}"
    for extension in [".cu", ".cuh", ".cpp", ".hpp", ".template.cu"]
    if os.path.exists(f"cuda/{base_filename}{extension}")
)
hip_filename = next(
    f"hip/{base_filename}{extension}"
    for extension in [".hip.cpp", ".hip.hpp", ".template.hip.cpp"]
    if os.path.exists(f"hip/{base_filename}{extension}")
)
output_filename = f"common/cuda_hip/{base_filename}{'.cpp' if cuda_filename.endswith('.cu') else '.hpp'}"

common_lines = list(open(common_filename))[3:]  # remove license header
cuda_lines = list(open(cuda_filename))
hip_lines = list(open(hip_filename))

cuda_file_guard = f"GKO_{cuda_filename.upper().replace('/', '_').replace('.','_')}_"
hip_file_guard = f"GKO_{hip_filename.upper().replace('/', '_').replace('.','_')}_"
common_file_guard = f"GKO_{common_filename.upper().replace('/', '_').replace('.','_')}_"

cuda_lines = [
    line.replace(cuda_file_guard, common_file_guard)
    .replace("namespace cuda", "namespace GKO_DEVICE_NAMESPACE")
    .replace("CudaExecutor", "DefaultExecutor")
    for line in cuda_lines
]
hip_lines = [
    line.replace(hip_file_guard, common_file_guard)
    .replace("namespace hip", "namespace GKO_DEVICE_NAMESPACE")
    .replace("HipExecutor", "DefaultExecutor")
    for line in hip_lines
]

for i in range(len(cuda_lines)):
    if cuda_lines[i].startswith('#include "'):
        cuda_lines[i] = (
            cuda_lines[i]
            .replace('#include "cuda/', '#include "common/cuda_hip/')
            .replace(".cuh", ".hpp")
            .replace("cublas", "blas")
            .replace("cusparse", "sparselib")
            .replace("curand", "randlib")
        )
    cuda_lines[i] = (
        cuda_lines[i]
        .replace("cuda_range", "device_range")
        .replace("cuda::", "GKO_DEVICE_NAMESPACE::")
    )
for i in range(len(hip_lines)):
    if hip_lines[i].startswith('#include "'):
        hip_lines[i] = (
            hip_lines[i]
            .replace('#include "hip/', '#include "common/cuda_hip/')
            .replace(".hip.hpp", ".hpp")
            .replace("hipblas", "blas")
            .replace("hipsparse", "sparselib")
            .replace("hiprand", "randlib")
        )
    hip_lines[i] = (
        hip_lines[i]
        .replace("hip_range", "device_range")
        .replace("hip::", "GKO_DEVICE_NAMESPACE::")
    )

cuda_location = next(
    i
    for i, line in enumerate(cuda_lines)
    if line.startswith(f'#include "{common_filename}"')
)
hip_location = next(
    i
    for i, line in enumerate(hip_lines)
    if line.startswith(f'#include "{common_filename}"')
)
cuda_replaced = (
    cuda_lines[:cuda_location] + common_lines + cuda_lines[cuda_location + 1 :]
)
hip_replaced = hip_lines[:hip_location] + common_lines + hip_lines[hip_location + 1 :]

cuda_replaced = (
    subprocess.run(
        args=[
            "/home/tribizel/.cache/pre-commit/repoay30okq9/py_env-python3/lib64/python3.9/site-packages/clang_format/data/bin/clang-format",
            f"-assume-filename={output_filename}",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=bytes("".join(cuda_replaced), "utf-8"),
    )
    .stdout.decode()
    .splitlines()
)
hip_replaced = (
    subprocess.run(
        args=[
            "/home/tribizel/.cache/pre-commit/repoay30okq9/py_env-python3/lib64/python3.9/site-packages/clang_format/data/bin/clang-format",
            f"-assume-filename={output_filename}",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=bytes("".join(hip_replaced), "utf-8"),
    )
    .stdout.decode()
    .splitlines()
)

if cuda_replaced == hip_replaced:
    with open(output_filename, "w") as file:
        file.write("\n".join(cuda_replaced))
    os.remove(common_filename)
    os.remove(cuda_filename)
    os.remove(hip_filename)
    with open("cuda_source_delete.sed", "a") as file:
        file.write("/" + cuda_filename[5:].replace("/", "\\/") + "/d;")
    with open("hip_source_delete.sed", "a") as file:
        file.write("/" + hip_filename[4:].replace("/", "\\/") + "/d;")
    with open("source_add.cmake", "a") as file:
        file.write(f"{output_filename}\n")
    sys.exit(0)
else:
    print(common_filename)
    print(cuda_filename)
    print(hip_filename)
    print("\n".join(difflib.unified_diff(cuda_replaced, hip_replaced)))
    sys.exit(1)
