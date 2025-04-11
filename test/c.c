#include <stdio.h>


void hello(char *str)
{
    printf("%s, %s\n", "hello", str);
}

typedef void (*funcModel)(char* str);

void just(void* func, char* str)
{
    ((funcModel)func)(str);
}


int main(int argc, char const *argv[])
{
    printf("start\n");
    just(hello, "shit");
    return 0;
}
