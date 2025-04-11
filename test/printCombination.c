#include<stdio.h>
#include<string.h>

void printArr(int *a, int len) {
    for (int i = 0; i < len; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");    
}

void printC(int n) {
    int chose[n];
    int chosen[n+1];
    int index = 0;
    int i = 1;
    int goback = 0;

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
            printArr(chose, n);  //
            chosen[i] = 0;
            index--;
        }
    }
}

int main(int argc, char const *argv[])
{
    printC(3);
    return 0;
}
