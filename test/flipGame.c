#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

void print2dArr(int rows, int cols, int arr[][cols])
{
    printf("   ");
    for (int i = 0; i < cols; i++)
    {
        printf("%d  ", i);
    }
    printf("\n");
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j == 0) {
                printf("%d  ", i);
            }
            printf("%d  ", arr[i][j]);
        }
        printf("\n");
    }
}

int click(int rows, int cols, int arr[rows][cols]) {
    int a, b;
    scanf("%d%d", &a, &b);
    if (a >= rows || b >= cols || a<0 || b<0) {
        return 0;
    }
    arr[a][b] ^= 1;
    if (a > 0) {
        arr[a-1][b] ^= 1;
    }
    if (b > 0) {
        arr[a][b-1] ^= 1;
    }
    if (a < rows - 1) {
        arr[a+1][b] ^= 1;
    }
    if (b < cols - 1) {
        arr[a][b+1] ^= 1;
    }
    print2dArr(rows, cols, arr);
    return 1;
}

// int xor(int a, int b) {
//     return (~(a&b)) & (a|b);
// }

int main(int argc, char const *argv[])
{
    // printf("%d, %d, %d, %d\n", xor(1, 0), xor(0, 0), xor(1, 1), xor(0, 1));
    int rows = 50;
    int cols = 50;
    printf("please input rows cows:");
    scanf("%d %d", &rows, &cols);
    int arr[rows][cols];
    srand((unsigned int)time(NULL));
    // printf("%ld\n", time(0)/24/3600/365 + 1970);
    for (size_t i = 0; i < rows; i++)
    {
        int r = rand();
        for (int j = 0; j < cols; j++)
        {
            arr[i][j] = r & 1;
            r = r >> 1;
        }
    }
    print2dArr(rows, cols, arr);
    while (click(rows, cols, arr));
    
    return 0;
}
