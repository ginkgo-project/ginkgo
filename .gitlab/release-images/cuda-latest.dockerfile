FROM nvidia/cuda:12.9.0-devel-ubuntu24.04
RUN apt update && apt install -y git cmake ninja-build python3
