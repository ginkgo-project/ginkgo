FROM archlinux/archlinux:latest
RUN pacman -Sy --noconfirm git gcc cmake ninja python3
