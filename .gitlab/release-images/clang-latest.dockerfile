FROM archlinux
RUN pacman -Sy --noconfirm git clang openmp cmake ninja python3
