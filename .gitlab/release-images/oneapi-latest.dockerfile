FROM intel/oneapi-hpckit:2025.1.3-0-devel-ubuntu24.04
RUN apt update && apt install -y ninja-build
