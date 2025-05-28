FROM archlinux/archlinux:latest
RUN pacman -Sy --noconfirm git clang openmp cmake ninja python3
