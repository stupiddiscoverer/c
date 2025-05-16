#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ulong unsigned long long 
#define uint unsigned int
#define log2_10 3.32192809488736 // log2(10) = 3.3219280948873626

typedef struct
{
    // 小端序存储更好计算
    uint size;
    uint lenth;
    uint *data; // uint max 42亿 可以存[0, 10亿) 9位十进制数字，但是乘法时，9亿*9 = 81亿 > 21亿，乘法可以用强转long乘法，每次乘法都要 /和%1亿 保留余数，除法结果做进位
    // 或者直接转全2进制，只是转换时麻烦点，算的时候很方便
    char sign; // 1表示负数，0表示正数
} BigNum;
BigNum* bigAdd(BigNum *bignum1, BigNum *bignum2);

void freeBigNum(BigNum *bignum)
{
    if (bignum != NULL)
    {
        free(bignum->data);
        free(bignum);
    }
}

BigNum* copyBigNum(BigNum *bignum)
{
    BigNum *result = (BigNum *)malloc(sizeof(BigNum));
    memcpy(result, bignum, sizeof(BigNum));
    result->data = (uint*)malloc(result->size * sizeof(uint));
    memcpy(result->data, bignum->data, result->size * sizeof(uint));
    return result;
}

void printHex(BigNum *bignum)
{
    if (bignum->sign == -1)
    {
        printf("-");
    }
    if (bignum->lenth == 0)
    {
        printf("0x0");
        return;
    }
    printf("0x%x", bignum->data[bignum->lenth-1]);
    for (int i = bignum->lenth-2; i >= 0; i--)
    {
        printf("%08x", bignum->data[i]);
    }
    printf("\n");
}
void bigDivSmall(BigNum* bignum, int smallNum) {
    
}

void printBigNum(BigNum *bignum)
{
    if (bignum->sign == -1)
    {
        printf("-");
    }
    if (bignum->lenth == 0)
    {
        printf("0\n");
        return;
    }
    BigNum *temp = copyBigNum(bignum);
    char *str = (char*)calloc(temp->lenth*32/log2_10 + 2, sizeof(char));
    int strIndex = temp->lenth*32/log2_10;
    ulong remainder = 0;
    while (temp->lenth>0)
    {
        for (int i = temp->lenth-1; i >= 0; i--)
        {
            remainder = remainder << 32 | temp->data[i];
            temp->data[i] = remainder / 1000000000;
            remainder = remainder % 1000000000;
            if (temp->data[i] == 0)
            {
                temp->lenth--;
            }
        }
        for (int i = 0; i < 9 && strIndex-i>=0; i++)
        {
            str[strIndex - i] = remainder % 10 + '0';
            remainder /= 10;
        }
        strIndex -= 9;
    }
    char *strStart = str;
    while (*strStart == 0 || *strStart == '0') {
        strStart++;
    }
    printf("%s\n", strStart);
    // printf("%s\n", str);
    freeBigNum(temp);
    free(str);
}

void bigMulSmall(BigNum *bignum, int smallNum)
{
    if (smallNum < 0) {
        bignum->sign = -bignum->sign;
        smallNum = -smallNum;
    }
    // 乘法
    uint carry = 0;
    for (int i = 0; i < bignum->lenth; i++)
    {
        ulong temp = (ulong)bignum->data[i] * smallNum + carry; 
        //比如 ff*ff + ff = fffe < ffff 相当于 99*99 + 99 = 9998 < 9999不会溢出
        bignum->data[i] = temp;
        carry = temp >> 32;
    }
    if (carry != 0)
    {
        bignum->data[bignum->lenth] = carry;
        bignum->lenth++;
    }
}

void bigAddSmall(BigNum *bignum, int smallNum)
{
    if (bignum->lenth == 0 && smallNum != 0)
    {
        bignum->lenth++;
    }
    // 加法
    int carry = 0;
    for (int i = 0; i < bignum->lenth; i++)
    {
        ulong temp = (ulong)bignum->data[i] + smallNum + carry;
        smallNum = 0; // 只加一次
        bignum->data[i] = temp;
        if (smallNum < 0 && bignum->data[i] < -smallNum) {
            carry = -1;
        } else {
            carry = temp >> 32;
        }
        if (carry == 0) {
            break;
        }
    }
    if (carry != 0)
    {
        bignum->data[bignum->lenth] = carry;
        bignum->lenth++;
    }
}

void setBigNumLenth(BigNum *bignum, int lenth)
{
    if (lenth > bignum->size)
    {
        bignum->size = lenth * 2;
        bignum->data = (uint*)realloc(bignum->data, bignum->size * sizeof(uint));
    }
    bignum->lenth = lenth;
}

BigNum* strToBigNum(char *num, int strlen)
{
    BigNum *bignum = (BigNum *)calloc(1, sizeof(BigNum));
    bignum->sign = 1;
    if (num[0] == '-')
    {
        bignum->sign = -1;
        num++;
        strlen--;
    }
    bignum->size = strlen*log2_10/32 + 1; // 向上取整
    bignum->data = (uint*)calloc(bignum->size, sizeof(uint));
    int i = 0;
    int temp = 0;
    while (i < strlen)
    {
        /* code */
        if (num[i] < '0' || num[i] > '9')
        {
            printf("error: invalid number\n");
            freeBigNum(bignum);
            return NULL;
        }
        temp = temp * 10 + (num[i] - '0');

        if (i%9 == 8) // 9位数最多是9.9亿
        {
            bigMulSmall(bignum, 1000000000); // 乘10亿
            bigAddSmall(bignum, temp);
            temp = 0;
        } else if (i == strlen - 1)
        {
            int x = i % 9;
            int y = 10;
            for (int j = 0; j < x; j++)
            {
                y *= 10;
            }
            bigMulSmall(bignum, y);
            bigAddSmall(bignum, temp);
        }
        i++;
    }
    if (bignum->lenth == 1 && bignum->data[0] == 0)
    {
        bignum->lenth = 0;
    }
    return bignum;
}

int bigCmp(BigNum *bignum1, BigNum *bignum2, int abs)
{
    int sign = bignum1->sign;
    if (abs == 0) { // 考虑符号位
        if (bignum1->sign != bignum2->sign)
        {
            return bignum1->sign - bignum2->sign;
        }
    } else { // 只比较绝对值
        sign = 1;
    }
    if (bignum1->lenth != bignum2->lenth)
    {
        return (bignum1->lenth - bignum2->lenth) * sign;
    }
    for (int i = bignum1->lenth-1; i >= 0; i--)
    {
        if (bignum1->data[i] != bignum2->data[i])
        {
            return (bignum1->data[i] > bignum2->data[i] ? 1 : -1) * sign;
        }
    }
    return 0;
}

BigNum* bigSub(BigNum *bignum1, BigNum *bignum2){
    if (bignum1->sign != bignum2->sign)
    {
        bignum2->sign *= -1;
        BigNum *ret = bigAdd(bignum1, bignum2);
        bignum2->sign *= -1;
        return ret;
    }
    BigNum* big = bignum1;
    BigNum* small = bignum2;
    if (bigCmp(bignum1, bignum2, 1) < 0)
    {
        big = bignum2;
        small = bignum1;
    }
    BigNum *result = (BigNum *)calloc(1, sizeof(BigNum));
    result->sign = big->sign;
    result->size = big->lenth;
    result->lenth = big->lenth;
    result->data = (uint*)calloc(result->size, sizeof(uint));
    int carry = 0;
    for (int i = 0; i < result->lenth; i++)
    {
        if (big->data[i] + carry < small->data[i]) {
            result->data[i] = big->data[i] + carry + ~small->data[i] + 1;
            carry = -1;
        }else {
            result->data[i] = big->data[i] + carry - small->data[i];
            carry = 0;
        }
    }
    while (result->lenth > 0 && result->data[result->lenth-1] == 0)
    {
        result->lenth--;
    }
    return result;
}
BigNum* bigAdd(BigNum *bignum1, BigNum *bignum2)
{
    if (bignum1->sign != bignum2->sign)
    {
        bignum2->sign *= -1;
        BigNum *ret = bigSub(bignum1, bignum2);
        bignum2->sign *= -1;
        return ret;
    }
    BigNum* big = bignum1;
    BigNum* small = bignum2;
    if (bignum1->lenth < bignum2->lenth)
    {
        big = bignum2;
        small = bignum1;
    }

    BigNum *result = (BigNum *)calloc(1, sizeof(BigNum));
    result->sign = big->sign;
    result->size = big->lenth + 1;
    result->lenth = big->lenth;
    result->data = (uint*)calloc(result->size, sizeof(uint));
    int carry = 0;
    for (int i = 0; i < big->lenth; i++)
    {
        if (i < small->lenth)
        {
            ulong temp = (ulong)big->data[i] + small->data[i] + carry;
            carry = temp >> 32;
            result->data[i] = temp;
        } else if (carry == 1){
            ulong temp = (ulong)big->data[i] + carry;
            carry = temp >> 32;
            result->data[i] = temp;
        } else {
            result->data[i] = big->data[i];
        }
    }
    if (carry != 0)
    {
        setBigNumLenth(result, result->lenth + 1);
    }
    return result;
}


BigNum* bigMul(BigNum *bignum1, BigNum *bignum2)
{
    BigNum *result = (BigNum *)calloc(1, sizeof(BigNum));
    if (bignum1->lenth == 0 || bignum2->lenth == 0)
    {
        result->sign = 1;
        result->lenth = 0;
        result->size = 1;
        result->data = (uint*)calloc(1, sizeof(uint));
        return result;
    }
    result->sign = bignum1->sign * bignum2->sign;
    result->size = bignum1->lenth + bignum2->lenth;
    result->lenth = bignum1->lenth + bignum2->lenth;
    result->data = (uint*)calloc(result->size, sizeof(uint));
    uint carry = 0;
    for (int i = 0; i < bignum1->lenth; i++)
    {
        for (int j = 0; j < bignum2->lenth; j++)
        {
            ulong temp = (ulong)bignum1->data[i] * bignum2->data[j] + result->data[i+j] + carry;  // (x - 1)^2  + 2(x-1) = x^2 - 1 < x^2 不会溢出
            result->data[i+j] = temp;
            carry = temp >> 32;
        }
        int k = i + bignum2->lenth;
        while (carry != 0)
        {
            ulong temp = (ulong)result->data[k] + carry;  // (x - 1)^2  + 2(x-1) = x^2 - 1 < x^2 不会溢出
            result->data[k++] = temp;
            carry = temp >> 32;
        }
    }
    if (result->data[result->lenth-1] == 0)
    {
        result->lenth--;
    }
    return result;
}

uint div96_64(uint pre, ulong end, ulong divsor) {
    uint divsor_top32 = divsor >> 32;
    if (divsor_top32 == 0)
    {
        printf("error: divsor is smaller than 2^32\n");
        return 0;
    }
    if (((ulong)pre << 32 | end >> 32) >= divsor)
    {
        printf("error: result is bigger than 2^32 -1 \n");
        return 0;
    }
    int pre0 = __builtin_clzl(divsor);
    divsor_top32 = divsor >> 32-pre0;
    ulong pre64 = (ulong)pre << 32+pre0 | end >> 32-pre0;
    uint top = pre64 / divsor_top32;
    uint bottom = pre64 / (ulong)divsor_top32 + 1; //divsor_top32 + 1可能溢出， divsor_top32较大时+1几乎没影响，较小时，pre也小,可以向后移动，保证 divsor_top32 >= 2^31,
                                            // 这样top-bottom的差距比例<= [(x/2^31) - (x/(2^31 +1))] / (x/2^31) = 1/(2^31+1) < 2^-31，
                                            // 而pre <= (divsor_top32)保证top<=2^32-1, 即使top=2^32-1，差距 (2^32-1) * 2^-31 = 2-2^-31, 即 0 <= top-bottom <= 1
    uint remain = pre64 - top * divsor_top32;
    uint divsor_low = divsor << 32+pre0 >> 32+pre0;
    uint end_low = end << 32+pre0 >> 32+pre0;
    if ((ulong)top * divsor_low - end_low > (ulong)remain << 32-pre0)
    {
        if (top == 0)
        {
            printf("error: result is smaller than 0\n");
            return 0;
        }
        return bottom;
    }
    return top;
}

BigNum* bigDiv(BigNum *bignum1, BigNum *bignum2, BigNum** remainderBig)
{
    if (bignum2->lenth == 0)
    {
        printf("error: divide by zero\n");
        return NULL;
    }
    BigNum* remainder = copyBigNum(bignum1);
    *remainderBig = remainder;
    BigNum *result = (BigNum *)malloc(sizeof(BigNum));
    if (bigCmp(bignum1, bignum2, 1) < 0)
    {
        result->sign = 1;
        result->lenth = 0;
        result->size = 1;
        result->data = (uint*)calloc(1, sizeof(uint));
        return result;
    } //123456789987654321123456789987654321 123456789987654321123456789
    result->sign = bignum1->sign * bignum2->sign;
    result->lenth = bignum1->lenth - bignum2->lenth + 1;
    result->size = result->lenth;
    result->data = (uint*)calloc(result->size, sizeof(uint));
    remainder->lenth = bignum2->lenth;
    if (bignum2->lenth > 1) {
        ulong divsor = (ulong)bignum2->data[bignum2->lenth-1] << 32 | bignum2->data[bignum2->lenth-2]; 
        uint remain = 0;
        for (int r_i = result->lenth - 1; r_i >= 0; r_i--)
        {
            int i1 = r_i - result->lenth + bignum1->lenth;
            ulong pre = (ulong)remainder->data[i1] << 32 | remainder->data[i1-1]; //需要3*32位(remain<<64)|...|...
            
            result->data[r_i] = div96_64(remain, pre, divsor); // 0 <= 余数 < 2^32-1   0<=result<=2^32-1

            uint carry = 0;
            for (int i = 0; i < bignum2->lenth; i++) // reminder -= result->data[r_i] * bignum2;
            {
                ulong temp = (ulong)bignum2->data[i] * result->data[r_i] + carry; // max= (2^32-1)*(2^32-1) + 0 = (2^64-2^33) >> 32 = 2^32-2 第一次
                                                                                 //  max= (2^32-1)*(2^32-1) + 2^32-2 = (2^64-2^32 - 1) >> 32 = 2^32-2 第二次
                uint temp1 = temp;
                carry = temp >> 32; // 所以carry <= 2^32 - 2
                int index = i1 - bignum2->lenth + i + 1;
                if (temp1 > remainder->data[index])
                {
                    carry++; // 应该是reminder->data[index+1]--;但carry+1不会溢出，且remainder->data[index+1]可能index+1超过size
                    remainder->data[index] += ~temp1 + 1; // ~temp1 + 1 = -temp1, 
                    //找了半天是这里出错了 remainder->data[index] += remainder->data[i] + ~temp1 + 1 右边i没写成index，
                } else {
                    remainder->data[index] -= temp1;
                }
            }
            if (carry == remain + 1) { // if reminder < 0, reminder = reminder + bignum2, result->data[r_i]--;
                result->data[r_i]--;
                carry = 0;
                for (int i = 0; i < bignum2->lenth; i++) {
                    int index = i1 - bignum2->lenth + i + 1;
                    ulong temp = remainder->data[index] + bignum2->data[i] + carry;
                    remainder->data[index] = temp;
                    carry = temp >> 32;
                } // carry必须=1
                if (carry != 1)
                {
                    printf("error: carry != 1\n");
                }
            }
            remain = remainder->data[i1];
            remainder->data[i1] = 0;
        }
    } else {
        ulong remain = 0;
        for (int r_i = result->lenth - 1; r_i >= 0; r_i--)
        {
            remain = (remain << 32) | remainder->data[r_i];
            result->data[r_i] = remain / bignum2->data[0];
            remain = remain % bignum2->data[0];
        }
        remainder->data[0] = remain;
    }
    while (remainder->lenth > 0 && remainder->data[remainder->lenth-1] == 0)
    {
        setBigNumLenth(remainder, remainder->lenth - 1);
    }
    if (result->lenth > 0 && result->data[result->lenth-1] == 0)
    {
        setBigNumLenth(result, result->lenth - 1);
    }
    return result;
}

void test() {
    int a = 123456789;
    int b = -123456789;
    
    printf("%d,%d\n", a/-5, a%-5); // 结果为负数，余数为正数
    printf("%d,%d\n", b/-5, b%-5); // 结果为正数，余数为负数 ， 说明余数和被除数符号相同
    printf("%d,%d\n", b/5, b%5); // 结果为正数，余数为负数 ， 说明余数和被除数符号相同
    printf("%d,%d\n", 5/b, 5%b); // 结果为正数，余数为负数 ， 说明余数和被除数符号相同
    uint ret = div96_64(123, 123456789987654321, 123456789987654321);
    printf("ret: %u\n", ret);   
}

int main(int argc, char const *argv[])
{
    // test();
    // exit(0);
    char num1[1024] = {0};
	char num2[1024] = {0};
	scanf("%s %s", num1, num2); // 输入两个大数
    
    BigNum *bignum1 = strToBigNum(num1, strlen(num1));
    printHex(bignum1);
    printBigNum(bignum1);
    BigNum *bignum2 = strToBigNum(num2, strlen(num2));
    BigNum *resultBig = bigAdd(bignum1, bignum2);
    printf("加法结果: ");
    printBigNum(resultBig);
    freeBigNum(resultBig);

    resultBig = bigSub(bignum1, bignum2);
    printf("减法结果: ");
    printBigNum(resultBig);
    freeBigNum(resultBig);

    resultBig = bigMul(bignum1, bignum2);
    printf("乘法结果: ");
    printBigNum(resultBig);
    freeBigNum(resultBig);

    BigNum* remainderBig = NULL;
    resultBig = bigDiv(bignum1, bignum2, &remainderBig);
    printf("除法结果: ");
    printBigNum(resultBig);
    printf("余数: ");
    printBigNum(remainderBig);
    return 0;
} // 无法通过测试用例 123456789987654321123456789987654321 1234567899876543211
