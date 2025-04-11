#include<stdio.h>

int func(int arr[4]) {
    arr[1] = 3;
}
int main(int argc, char const *argv[])
{
    int ar[] = {1, 2, 3, 4};
    func(ar);
    printf("%d\n", ar[1]);
    return 0;
}
