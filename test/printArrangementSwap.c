#include <stdio.h>

// 函数交换两个元素的位置
void swap1(int *a, int *b) {
    *a = *a ^ *b;
    *b = *a ^ *b;
    *a = *a ^ *b;
}

void swap(int*a, int*b) {
    int temp =*a;
    *a = *b;
    *b = temp;
}

// 递归生成并打印所有排列
void permute(int arr[], int start, int n) {
    // 当起始位置等于 n 时，说明排列已经生成完毕
    if (start == n) {
        // 打印当前的排列
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        return;
    }

    // 递归生成排列
    for (int i = start; i < n; i++) {
        // 交换元素，固定第 start 个元素
        swap(&arr[start], &arr[i]);

        // 递归生成剩余元素的排列
        permute(arr, start + 1, n);

        // 回溯：交换回来，恢复原来的排列
        swap(&arr[start], &arr[i]);
    }
}

int main() {
    int n;

    // 输入 n 的值
    printf("Enter a number: ");
    scanf("%d", &n);

    // 创建一个数组来存储 1 到 n 的数字
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }

    // 打印所有排列
    permute(arr, 0, n);
    return 0;
}