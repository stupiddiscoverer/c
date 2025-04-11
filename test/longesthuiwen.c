#include<stdio.h>
#include<string.h>

void printStr(char* a, int len) {
    for(int i=0; i<len; i++) {
        printf("%c", a[i]);
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{
    char a[1000];
    scanf("%s", a);
    int len = strlen(a);
    int maxHalfLenOdd = 1;
    int oddMid = 0;
    int maxHalfLenEven = 0;
    int evenMid = 0;

    for (int i=0; i<len; i++) {
        int j = 1;

        while (i+j < len && i-j>=0 && a[i+j] == a[i-j])
        {
            j++;
        }
        if (maxHalfLenOdd < j-1) {
            maxHalfLenOdd = j-1;
            oddMid = i;
        }
        j = 1;
        while (i+j < len && i-j+1>=0 && a[i+j] == a[i-j+1])
        {
            j++;
        }
        if (maxHalfLenEven < j-1) {
            maxHalfLenEven = j-1;
            evenMid = i;
        }
    }
    if (maxHalfLenOdd > maxHalfLenEven) {
        printStr(&a[oddMid - maxHalfLenOdd], 2*maxHalfLenOdd+1);
    }
    else {
        printStr(&a[evenMid - maxHalfLenEven + 1], 2*maxHalfLenEven);
    }
    // printf("%d\n", len);
    return 0;
}
