#include <stdio.h>
#include <stdint.h>

int main() {
    double x = 64.0;
    int y = 64;
    uint64_t *p = (uint64_t *)&x;
    printf("十六进制存储: 0x%016lX\n", *p);
    printf("十六进制存储: 0x%08lX\n", y);
    printf("x == (int)x: %d\n", x == (int)x);
    printf("*(int*)&x: %d\n", *(int*)&x);
    return 0;
}