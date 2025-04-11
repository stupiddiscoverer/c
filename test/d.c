#include <stdio.h>
#include <stdlib.h>

/**
 * 题目是给资源 A B C的数量和 n行产品所需资源及对应回报
 * 
 * 例子 n=3 a=7 b=8 c=9
 * 1. 需要资源  2 2 3  回报 2
 * 2. 需要资源  2 4 2  回报 3
 * 3. 需要资源  5 4 1  回报 3
 * 求最多能得多少回报
 * 这题显然只能选1，2产品，回报是5
*/
typedef struct d
{
    int index;
    int value;
} iv;

int sum_array_column(int choose[][4], int row)
{ 
    int sum = 0;
    for (int i = 0; i < row; i++)
    {
        sum += choose[i][3];
    }
    return sum;
}

void print2DArr(int *a, int rows, int columns) {
    for (size_t i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("%d ", a[i*columns + j]);
        }
        printf("\n");
    }
    
}

int calculate_maxvalue(int str[][4], int n, int A, int B, int C){
    return 0;
}

int clean(int *a[4], int n, int A, int B, int C, int *newArr[4]) {
    int k = 0;
    for (size_t i = 0; i < n; i++)
    {
        
        if(a[i][4] <= 0 || a[i][0] > A || a[i][1] > B || a[i][2] > C) {
            continue;
        }
        for (size_t j = 0; j < 4; j++)
        {
            newArr[k][j] = a[i][j];
        }
        k++;
    }
    return k;
}

void sort(int *arr[4], int *rank, int n) {
    int a[n][3];
    printf("%d", a[0][2]);
}




int main()
{
    int n, A, B, C;
    scanf("%d%d%d%d", &n, &A, &B, &C);
    int str[n][4];
    int newstr[n][4];
    iv Asort[n];
    iv Bsort[n];
    iv Csort[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d%d%d%d", &str[i][0], &str[i][1], &str[i][2], &str[i][3]);
        copy1[i].index = i;
        copy1[i].value = str[i][3];
    }
    print2DArr((int*)str, n, 4);
    sort(copy1, copy2, n);
    // clean(str, n, A, B, C, newstr);
    // print2DArr(str, n);

    int max_value = calculate_maxvalue(newstr, n, A, B, C);
    printf("%d", max_value);                   
    return 0;
}