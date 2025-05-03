#include <stdio.h>
#include <string.h>

// 首先要保证能达到最大关卡，如果可以则要保证最终结果人数✖ 武器的值最大
// 如果一正一负一定选正数，如果都是武器或都是人则选最大的，没有疑问

// 最多7种情况，0-6个武器，假使人口优先仍然有x个武器，武器优先最后也只有y个武器，则情况减少到x-y
// 最重要的是记住每次选择优先武器的历史步骤直到满足指定武器数量，
// 如果当前步骤无法通关，说明历史步骤里的一些需要改回人口优先，优先改动人口减少多的步骤！
// 每次遇到可以选武器时比较历史步骤里人口减少量，如果当前步骤使得人口变多则替换历史步骤
int max(int a, int b) {
    return a>b?a:b;
}
int min(int a, int b) {
    return a<b?a:b;
}
int max3(int a, int b, int c) {
    return max(max(a, b), c);
}
int addWeapon(int* a, int b) {
    *a = min(*a+b, 6), 0;
    if (*a > 6) {
        *a = 6;
        return 1;
    }
    return 0;
}
int maxPower(int n, int choices[n][2]) {
    int people = 1;
    int weapon = 0;
    int doubt[n];
    memset(doubt, -1, sizeof(doubt));
    int doubtStart = 0;
    int weaponOverflow = 0;
    int trace[7][7]; // 7种情况，7条武器优先的历史记录？武器优先n次一定使得当前武器比人口优先多n个，否则只多m个说明之前有n-m次选择白选了？不见得，可能要选10次才可以多3个武器，因为中间可能必须选择7次减少武器
    for (int i = 0; i < n; i++)
    {
        int leftPeo = 0, rightPeo = 0;
        int leftWea = 0, rightWea = 0;
        if (choices[i][0] == 1000 || choices[i][0] == -1000) {
            leftPeo = 0;
            leftWea = choices[i][0] / 1000;
        } else {
            leftPeo = choices[i][0];
        }
        if (choices[i][1] == 1000 || choices[i][1] == -1000) {
            rightPeo = 0;
            rightWea = choices[i][0] / 1000;
        } else {
            rightPeo = choices[i][1];
        }
        if ((leftWea * rightPeo > 0) || (leftPeo * rightWea > 0)) {
            // 既有武器又有人，且武器和人同正负，选项存疑
            if (leftPeo < 0 || leftWea < 0) {
                // 若同为负数，先选武器，保证可以通关
                weaponOverflow = addWeapon(&weapon, min(leftWea, rightWea));
                if (people + min(leftPeo, rightPeo) > 0) {
                    // 2个选项都可以，所以存疑
                    doubt[i] = leftWea==0?1:0;
                } else {
                    // 选人导致不通关，没有疑问，肯定选武器，并且之前的选择也没有疑问，必须保证人数
                    doubtStart = i+1;
                }
            } else {
                // 选人，保证可以通关
                people += max(leftPeo, rightPeo);
                doubt[i] = leftPeo==0?1:0;
            }
        } else {
            // 全是武器，或全是人，或武器和人不同号
            int maxPeo = max(leftPeo, rightPeo);
            int maxWea = max(leftWea, rightWea);
            if (leftWea * rightWea == 0) {
                // 有1或0个武器
                if (maxWea == 0) {  
                    // 若有1个武器，则该武器是-1，则另一个人>=0，或全是人
                    people += maxPeo;
                } else {
                    // 武器是1， 则人<=0
                    weaponOverflow = addWeapon(&weapon, maxWea);
                }
            } else {
                // 都是武器
                weaponOverflow = addWeapon(&weapon, maxWea);
            }
        }
        if (weaponOverflow) {
            doubtStart = i+1; // 之前的选择没有疑问，即使尽量选人，使得人数最大化了，武器依然溢出
        }
    }
    // 接下来从doubt中穷举可能选择？动态规划应该是选择时穷举可能选择，因为每次选择会影响后面
    return 0;
}

int canWin(int n, int choices[n][2]) {
    int people = 1;
    for (int i = 0; i < n; i++)
    {
        int leftPeo = 0, rightPeo = 0;
        if (choices[i][0] == 1000 || choices[i][0] == -1000) {
            leftPeo = 0;
        } else {
            leftPeo = choices[i][0];
        }
        if (choices[i][1] == 1000 || choices[i][1] == -1000) {
            rightPeo = 0;
        } else {
            rightPeo = choices[i][1];
        }
        people += leftPeo > rightPeo ? leftPeo : rightPeo;
        if (people <= 0) {
            return i+1;
        }
    }
    return 0;
}

int main() {
    // printf("Hello, World!\n");
    int n = 0;
    scanf("%d", &n);
    int choice[n][2];
    for (int i = 0; i < n; i++)
    {
        scanf("%d %d", &choice[i][0], &choice[i][1]);
    }
    int result = canWin(n, choice);
    if (result == 0) {
        printf("1 ");
    } else {
        printf("0 %d\n", result);
        return 0;
    }
    result = maxPower(n, choice);
    printf("%d\n", result);
    return 0;
}