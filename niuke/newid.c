#include <stdio.h>

int main() {
    long long x, y;
    while (scanf("%lld %lld", &x, &y) != EOF) { // 注意 while 处理多个 case
        // 64 位输出请用 printf("%lld") to 
        long pre = 1;
        for (int i=0; i<y; i++) {
            pre *= 26;
        }
        x = (x+pre-1) / pre;
        int z = 0;
        long end = 1;
        while (x > end) {
            end *= 10;
            z++;
        }
        if (z==0) {
            z=1;
        }
        printf("%d\n", z);
    }
    return 0;
}