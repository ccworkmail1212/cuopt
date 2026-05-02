// glibc_compat.c
// 讓 Ubuntu 24.04 build 的 libcuopt.so 能在 Ubuntu 22.04 (GLIBC 2.35) 運行
// Ubuntu 24.04 的 GLIBC 2.38 把 fscanf/strtol 重定向到 __isoc23_* 變體
// 這個 shim 在 GLIBC 2.35 系統上提供這些 symbols，委派給標準函式

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

int __isoc23_fscanf(FILE *stream, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int ret = vfscanf(stream, format, args);
    va_end(args);
    return ret;
}

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}
