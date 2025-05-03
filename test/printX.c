#include <stdio.h>
#include <stdint.h>

int main() {
    double x = 64;
    int y = 64;
    uint64_t *p = (uint64_t *)&x;
    printf("十六进制存储: 0x%08X\n", y);
    printf("x == (int)x: %d\n", x == (int)x);
    printf("*(int*)&x: %d\n", *(int*)&x);
    int a = 999;
    int b = -23;
    printf("%d/%d=%d......%d\n", a, b, a / b, a % b);
    printf("%d * %d=%d + %d = %d\n", b, a/b, b*(a/b), a%b, b * (a/b) + a % b);
    void* c[2];
    printf("%ld, %p, %ld\n", (long)c, c, (long)c[0]);
    printf("十六进制存储: 0x%016lx\n", (long)c);

    int yy = 0x12345678;
    char *yp = (char *)&yy;
    for (int i = 0; i < sizeof(y); i++) {
        printf("%02x ", yp[i]);
    }
    printf("\n");
    return 0;
}