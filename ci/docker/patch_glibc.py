#!/usr/bin/env python3
"""Patch libcuopt.so to replace GLIBC_2.38 requirement with GLIBC_2.17
so it can run on Ubuntu 22.04 (GLIBC 2.35) / official cuopt image.
"""
import struct, sys, shutil

def elf_hash(name):
    h = 0
    for c in name.encode():
        h = ((h << 4) + c) & 0xffffffff
        g = h & 0xf0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xffffffff
    return h

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src

with open(src, 'rb') as f:
    data = bytearray(f.read())

# Parse ELF64 header
assert data[:4] == b'\x7fELF', "Not ELF"
e_shoff     = struct.unpack_from('<Q', data, 40)[0]
e_shentsize = struct.unpack_from('<H', data, 58)[0]
e_shnum     = struct.unpack_from('<H', data, 60)[0]
e_shstrndx  = struct.unpack_from('<H', data, 62)[0]
shstr_off   = struct.unpack_from('<Q', data, e_shoff + e_shstrndx * e_shentsize + 24)[0]

# Patch map: version string in libcuopt.so → compatible version in official image (GLIBC 2.35)
# GLIBC_2.38  → GLIBC_2.17   (Ubuntu 24.04 C runtime new ISO C23 symbols)
# GLIBCXX_3.4.31 → GLIBCXX_3.4.30  (GCC 13 libstdc++ vs GCC 12 in official image)
PATCHES = [
    (b'GLIBC_2.38',     b'GLIBC_2.17'),    # same length: 10 chars
    (b'GLIBCXX_3.4.31', b'GLIBCXX_3.4.30'), # same length: 11 chars
]
old_name, new_name = PATCHES[0]  # primary (overridden in loop)
verneed_off = verneed_size = strtab_off = 0
for i in range(e_shnum):
    sh = e_shoff + i * e_shentsize
    name_off = struct.unpack_from('<I', data, sh)[0]
    sh_type   = struct.unpack_from('<I', data, sh + 4)[0]
    sh_offset = struct.unpack_from('<Q', data, sh + 24)[0]
    sh_size   = struct.unpack_from('<Q', data, sh + 32)[0]
    name = data[shstr_off + name_off:shstr_off + name_off + 20].split(b'\x00')[0]
    if name == b'.gnu.version_r':
        verneed_off, verneed_size = sh_offset, sh_size
    elif name == b'.dynstr':
        strtab_off = sh_offset

print(f'.gnu.version_r @ 0x{verneed_off:x}, .dynstr @ 0x{strtab_off:x}')

total_patched = 0
for old_name, new_name in PATCHES:
    old_hash = elf_hash(old_name.decode())
    new_hash = elf_hash(new_name.decode())
    print(f'{old_name.decode()} (0x{old_hash:08x}) → {new_name.decode()} (0x{new_hash:08x})')
    off = verneed_off
    while off < verneed_off + verneed_size:
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from('<HHIII', data, off)
        aux_off = off + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from('<IHHII', data, aux_off)
            ver = data[strtab_off + vna_name:strtab_off + vna_name + 20].split(b'\x00')[0]
            if ver == old_name:
                print(f'  found @ aux 0x{aux_off:x}, strtab+{vna_name}')
                struct.pack_into('<I', data, aux_off, new_hash)
                idx = strtab_off + vna_name
                data[idx:idx + len(old_name)] = new_name
                total_patched += 1
            if vna_next == 0:
                break
            aux_off += vna_next
        if vn_next == 0:
            break
        off += vn_next

print(f'Total patched: {total_patched} entries')
with open(dst, 'wb') as f:
    f.write(bytes(data))
print(f'Written to {dst}')
