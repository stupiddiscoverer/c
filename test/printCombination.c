#include<stdio.h>
#include<string.h>

void printArr(int *a, int len) {
    for (int i = 0; i < len; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");    
}

void printPermutation(int n) {
    int chose[n];
    int chosen[n+1];
    int index = 0;
    int i = 1;

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

void printCombination(int n, int *arr, int index, int start, int end) {
    if (index == n) {
        printArr(arr, n);
        return;
    }
    for (int i = start; i <= end && end - i + 1 >= n - index; i++) {
        arr[index] = i;
        printCombination(n, arr, index + 1, i + 1, end);
    }
}

void C(int numToSelect, int *nList, int m, int i, int index){
    if (numToSelect<1) {
        printArr(nList, numToSelect+index);
        return;
    }
    while (i<m) {
        nList[index] = i;
        i += 1;
        C(numToSelect-1,nList, m, i, index+1);  // 保持n + index = n
    }
}

void P(int numToSelect, int *nList, int m, int i, int index){
    if (numToSelect<1) {
        printArr(nList, numToSelect+index);
        return;
    }
    while (i<m) {
        nList[index] = i;
        i += 1;
        P(numToSelect-1,nList, m, i, index+1);  // 保持n + index = n
    }
}

int main(int argc, char const *argv[])
{
    printf("printPermutation\n");
    printPermutation(3);
    int arr[5] = {1, 2, 3, 4, 5};
    int n = 3;
    int len = sizeof(arr) / sizeof(arr[0]);

    printf("printCombination(%d, %d):\n", len, n);
    printCombination(n, arr, 0, 0, len - 1);
    printf("组C(%d, %d):\n", len, n);
    int nList[n];
    C(n, nList, len, 0, 0);
    return 0;
}
