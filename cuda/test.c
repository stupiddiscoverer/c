#include <stdio.h>

char add_with_overflow(int a, int b, int* result) {
    char overflow;
    asm (
        "addl %2, %1\n"
        "seto %0"
        : "=r"(overflow), "=r"(*result)
        : "r"(a), "1"(b)
        : "cc"
    );
    return overflow;
}

int main(int argc, char const *argv[])
{
    int result, a=(unsigned int)-1 >> 1, b=1;
    printf("a: %d, b: %d\n", a, b);
    char overflow = add_with_overflow(a, b, &result);
    printf("%d + %d = %d\n", a, b, result);
    if (overflow) { /* 溢出 */ 
        printf("Overflow occurred!\n");
    } else {
        printf("No overflow.\n");
    }

    b = 8;
    a = a + b;
    printf("a: %d, b: %d\n", (a<< 1) >> 1, b);
}
