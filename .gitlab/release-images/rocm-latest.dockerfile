FROM rocm/dev-ubuntu-24.04:6.4.1-complete
RUN apt update && apt install -y git cmake ninja-build python3
