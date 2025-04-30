#include<stdio.h>
#include<string.h>

typedef struct {
    int a;
    int b;
} number;


int main(int argc, char const *argv[])
{
    int a[100];
    number n = {0};
    memset(a, -1, sizeof(a));
    memset(&n, -1, sizeof(n));
    int b = 0;
    for(int i=0; i<100; i++) {
        // printf("%d ", a[i]);
        b+=a[i];
    }
    printf("%d, %d, %ld, %ld, %d\n", b, *(char*)&n, *(long*)&n, sizeof(a), *(char*)&b);
}