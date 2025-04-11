#include <stdio.h>

int readLine(char* str, int len) {
    int length = 0;
    while (length < len-1 && scanf("%c", &str[length]) == 1) {
        length++;
        if (str[length-1] == '\n') {
            break;
        }
    }
    return length;
}

int isdigit(char c) {
    return c <= '9' && c >= '0';
}

int isOperator(char c) {
    if (c == '+' || c == '-')
        return 1;
    if (c == '*' || c == '/')
        return 2;
    return 0;
}

int numInStr(char* str, int start) {
    int num = 0;
    while (isdigit(str[start]))
    {
        num *= 10;
        num += str[start] - '0';
    }
    return num;
}

int calculateMathStr(char* str, int start, int end) {
    
}

int findAllExpression(char* str, int len, int indexes[][2]) {
    int digit = 0;
    int operator = 0;
    char *c = str;
    char *end = &str[len-1];
    int index = 0;
    int start = 0;
    while (c <= end)
    {
        if (isdigit(*c)) {
            if (start == 0) {
                start = 1;
            }
            if (operator == '-') {

            }
        }
        c++;
    }
    
}

int main() {
    char str[501] = {0};
    int length = 0;
    int mathStrIndex[100][2] = {0};
    
    length = readLine(str, 500);
    printf("%s, %d\n", str, length);
    findAllExpression(str, length, mathStrIndex);
    
    return 0;
}