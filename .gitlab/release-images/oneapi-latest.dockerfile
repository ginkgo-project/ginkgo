FROM intel/oneapi-hpckit:2025.3.0-0-devel-ubuntu24.04
RUN apt update && apt install -y ninja-build
