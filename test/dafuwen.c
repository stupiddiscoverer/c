#include<stdio.h>
#include<string.h>

void printArr(int *a, int len) {
    for (int i = 0; i < len; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");    
}

void getComb(int n, int combs[][n]) {
    int chose[n];
    int chosen[n+1];
    int index = 0;
    int i = 1;
    int index_c = 0;

    memset(chose, 0, sizeof(chose));
    memset(chosen, 0, sizeof(chosen));

    while (index >= 0)
    {
        i = chose[index] + 1;
        while (i < n+1)   // 选一个数
        {
            if (chosen[i] == 0) {
                break;
            }
            i++;
        }
        if (i == n+1) {   // 无可选，回退
            chosen[chose[index]] = 0;
            chose[index] = 0;
            index--;
            continue;
        }

        chosen[chose[index]] = 0;   // 选择新数前，释放老数
        chose[index] = i;
        chosen[i] = 1;

        index++;
        if (index == n) {
            // printArr(chose, n);
            memcpy(combs[index_c++], chose, sizeof(chose));
            chosen[i] = 0;
            index--;
        }
    }
}

long long getSocre(int *arr, int choice[4], int len, int *legal) {
    long long score = 0;
    arr--;  //初始位置不在arr[0]。。在arr[-1]
    int *end = arr + len;
    *legal = 1;
    for (int i = 0; arr != end; i++)
    {
        arr += choice[i];
        if (arr > end) {
            *legal = 0;  //代表不是正确路径，无法到达n，返回的-1无效
            return -1;
        }
        score += *arr;
    }
    return score;
}

//在任意时刻，小美的金币数量都必须大于等于 0
long long maxSocre(int arr[], int len) {
    int combiantions[4*3*2][4] = {0};
    getComb(4, combiantions);
    // printArr(combiantions[3], 4);
    long long totalSocre = 0;
    long long maxSocre = 0;
    int legal = 1;

    for (int j=0; j<(len-1)/10 + 1; j++) {
        int x = 10;
        if (len - j*10 < x) {
            x = len - j*10;
        }
        int first = 1;
        for (int i=0; i<4*3*2; i++) {
            long long score = getSocre(arr+j*10, combiantions[i], x, &legal);
            if (!legal) {
                continue;
            }
            // printf("%d  ", first);
            if (first) {
                maxSocre = score;
                first = 0;
            }
            if (maxSocre < score) {
                maxSocre = score;
            }
        }
        totalSocre += maxSocre;
        if (totalSocre < 0) {
            return -1;
        }
    }
    return totalSocre;
}

int main(int argc, char const *argv[])
{
    int n;
    scanf("%d", &n);
    int arr[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
    }
    // printArr(arr, n);
    long long score = maxSocre(arr, n);
    printf("%lld\n", score);
    return 0;
}
