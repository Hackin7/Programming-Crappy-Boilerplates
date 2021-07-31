#!/bin/sh

# https://wiki.osdev.org/POSIX-UEFI
# Make sure to install lld before running
OVMF_PATH=/usr/share/ovmf/x64/OVMF.fd

#git clone https://gitlab.com/bztsrc/posix-uefi.git
#cd (your project)
#ln -s posix-uefi/uefi
make
rm *.o
rm uefi/*.o

#qemu-system-x86_64 -pflash bios.bin

uefi-run main.efi --bios $OVMF_PATH -q /bin/qemu-system-x86_64 #-- <extra_qemu_args>
