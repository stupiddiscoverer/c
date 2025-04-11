#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct  {
    char discribe[30];
    long size;
} capcity;

void countSize(capcity *ca) {
    long num = 0;
    int size = 0;
    for (int i=0; i<30; i++) {
        if (ca->discribe[i] >= '0' && ca->discribe[i] <= '9') {
            num = num*10 + ca->discribe[i] - '0';
        } else {
            if (ca->discribe[i] == 'T') {
                ca->size += num * 1024 * 1024;
            }
            if (ca->discribe[i] == 'G') {
                ca->size += num * 1024;
            }
            if (ca->discribe[i] == 'M') {
                ca->size += num;
            }
            num = 0;
            if (ca->discribe[i] == 0) {
                break;
            }            
        }
    }
}

void insertSort(capcity *cps, int len) {
    capcity temp = {0};
    for (int i=1; i<len; i++) {
        for (int j=0; j<i; j++) {
            if (cps[i].size > cps[j].size) {
                memcpy(&temp, &cps[i], sizeof(capcity));
                for (int k=i; k>j; k--) {
                    memcpy(&cps[k], &cps[k-1], sizeof(capcity));
                }
                memcpy(&cps[j], &temp, sizeof(capcity));
            }
        }
    }
}

int main() {
    int n = 0;
    while (scanf("%d", &n) != EOF) { // 注意 while 处理多个 case
        // 64 位输出请用 printf("%lld") to 
        capcity *drives = malloc(sizeof(capcity) * n);
        memset(drives, 0, n * sizeof(capcity));
        for (int i=0; i<n; i++) {
            scanf("%s", drives[i].discribe);
            countSize(&drives[i]);
        }
        insertSort(drives, n);
        for (int i=0; i<n; i++) {
            printf("%s\n", drives[i].discribe);
        }
    }
    return 0;
}