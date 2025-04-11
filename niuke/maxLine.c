#include <stdio.h>
#include <stdlib.h>

void printArr(char* arr, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            printf("%d  ", arr[i*cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int rows, cols;
    while (scanf("%d,%d", &rows, &cols) != EOF) { // 注意 while 处理多个 case
        // printf("\n%d,  %d\n", rows, cols);
        // 64 位输出请用 printf("%lld") to
        char* rectangle = malloc(rows * cols * sizeof(char));
        char* line = malloc(cols);
        for (int i = 0; i < rows; i++) {
            scanf("%s", line);
            for (int j = 0; j < cols; j++) {
                if (line[j * 2] == 'F') {
                    rectangle[cols * i + j] = 0;
                } else {
                    rectangle[cols * i + j] = 1;
                }
            }
        }
        // printArr(rectangle, rows, cols);

        int max = 0;
        for (int i = 0; i < rows; i++) {
            int tmpLen = 0;
            for (int j = 0; j < cols; j++) {
                if (j==0) {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
                if (rectangle[i * cols + j] == 1) {
                    tmpLen++;
                } else {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
            }
            if (tmpLen > max) {
                max = tmpLen;
            }
        }
        for (int i = 0; i < cols; i++) {
            int tmpLen = 0;
            for (int j = 0; j < rows; j++) {
                if (j==0) {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
                if (rectangle[j * cols + i] == 1) {
                    tmpLen++;
                } else {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
            }
            if (tmpLen > max) {
                max = tmpLen;
            }
        }
        for (int i = 1; i < rows + cols - 2; i++) {
            int tmpLen = 0;
            int x, y;
            if (i < cols) {
                x = 0;
                y = i;
            } else {
                x = i - cols + 1;
                y = cols - 1;
            }
            for (int j = 0; ; j++) {                
                if (x + j >= rows || y - j < 0) {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                    break;
                }
                if (rectangle[(x + j) * cols + (y - j)] == 1) {
                    tmpLen++;
                } else {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
            }
        }
        for (int i = 1; i < rows + cols - 2; i++) {
            int tmpLen = 0;
            int x, y;
            if (i < cols) {
                x = 0;
                y = cols - 1 - i;
            } else {
                x = i - cols + 1;
                y = 0;
            }
            for (int j = 0; ; j++) {                
                if (x + j >= rows || y + j >= cols) {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                    break;
                }
                if (rectangle[(x + j) * cols + (y + j)] == 1) {
                    tmpLen++;
                } else {
                    if (tmpLen > max) {
                        max = tmpLen;
                    }
                    tmpLen = 0;
                }
            }
        }
        printf("%d\n", max);
    }
    return 0;
}