#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct
{
	// 小端序存储更好计算
	int size;
	int lenth;
	char *data;
	char sign; // 1表示负数，0表示正数
} BigNum;

void reverseAndMinus(char *str, int len, int minus)
{
    if (len == 0) {
        return;
	}
	for (int i = 0; i < len / 2; i++)
	{
		char temp = str[i];
		str[i] = str[len - i - 1] - minus;
		str[len - i - 1] = temp - minus;
	}
    if (len % 2 == 1)
    {
        str[len / 2] -= minus; // 奇数长度时，中间的数字也要减去
    }
}

void compleMent(BigNum* num) {
	int carry = 1;
	for (int i = 0; i < num->lenth; i++)
	{
		num->data[i] = 9 - num->data[i] + carry; // 取反
		if (num->data[i] > 9)
		{
			num->data[i] -= 10;
			carry = 1;
		}
		else
		{
			carry = 0;
		}
	}
}
void changeSign(BigNum *num)
{
	num->sign = !num->sign; // 取反
	compleMent(num); // 取补码
}

char* bignum2char(BigNum *num)
{
	if (num->sign == 1) // 负数
	{
		compleMent(num); // 取补码
	}
	reverseAndMinus(num->data, num->lenth, -'0'); // 反转
	if (num->sign == 1) // 负数
	{
		for (int i = num->lenth; i > 0; i--)
		{
			num->data[i] = num->data[i - 1]; // 移动数据
		}
		num->data[0] = '-'; // 添加负号
	}
	return num->data;
}

// 初始化大数
BigNum *bignum_create(char *num, int size)
{
	BigNum *bignum = (BigNum *)malloc(sizeof(BigNum));
	memset(bignum, 0, sizeof(BigNum));
	bignum->data = num;
	bignum->size = size;
	bignum->lenth = strlen(num);
	if (num[0] == '-')
	{
		bignum->lenth--;
		memcpy(num, num + 1, bignum->lenth); // 去掉负号
		bignum->sign = 1;
	}
	for (int i = 0; i < bignum->lenth; i++) // 检查是否是数字
	{
		if (num[i] < '0' || num[i] > '9')
		{
			printf("Invalid number format\n");
			free(bignum);
			return NULL;
		}
	}
	reverseAndMinus(num, bignum->lenth, '0');  //小端存储，负数取补码
	if (bignum->sign == 1) // 负数取补码
		compleMent(bignum); // 负数取补码
	memset(num + bignum->lenth, 0, size - bignum->lenth); // 清空多余的部分
	return bignum;
}
// 大数加法 num1 = num1+num2 减法就是把被减数sign取反 sign=!sign
void bignum_add(BigNum *result, BigNum *num1, BigNum *num2)
{
	int maxlen = num1->lenth;
	// int sign = num1->sign;
	if (num1->lenth < num2->lenth) {
		maxlen = num2->lenth;
		// sign = num2->sign;
	}
	for (int i=num1->lenth; i <= maxlen; i++)
	{
		num1->data[i] = 9*num1->sign; // 补齐长度
	}
	for (int i=num2->lenth; i <= maxlen; i++)
	{
		num2->data[i] = 9*num2->sign; // 补齐长度
	}
	int carry = 0;
	for (int i = 0; i <= maxlen; i++)
	{
		result->data[i] = num1->data[i] + num2->data[i] + carry;
		if (result->data[i] > 9)
		{
			result->data[i] -= 10;
			carry = 1;
		}
		else
		{
			carry = 0;
		}
	}
	if (result->data[maxlen] > 1)
    {
        result->sign = 1; // 结果为负数
    }
    else
    {
        result->sign = 0; // 结果为正数
    }
    for (int i=num1->lenth; i <= maxlen; i++)
	{
		num1->data[i] = 0;
	}
	for (int i=num2->lenth; i <= maxlen; i++)
	{
		num2->data[i] = 0;
	}
    while (maxlen >= 0 && result->data[maxlen] == 9*result->sign)
    {
        result->data[maxlen] = 0; // 清除多余的部分
        maxlen--;
    }
    result->lenth = maxlen+1;
}

void bignum_sub(BigNum *result, BigNum *num1, BigNum *num2) {
	changeSign(num2); // 取反
	bignum_add(result, num1, num2); // 调用加法
	changeSign(num2); // 取反
}

void bignum_mul(BigNum *result, BigNum *num1, BigNum *num2) {
	result->sign = num1->sign ^ num2->sign; // 结果符号
	if (num1->sign == 1) // 负数取补码
		compleMent(num1);
	if (num2->sign == 1) // 负数取补码
		compleMent(num2);
	for (int i = 0; i < num1->lenth; i++)
	{
		for (int j = 0; j < num2->lenth; j++)
		{
			result->data[i + j] += num1->data[i] * num2->data[j];
			if (result->data[i + j] > 9)
			{
				result->data[i + j + 1] += result->data[i + j] / 10;
				result->data[i + j] %= 10;
			}
		}
	}
	result->lenth = num1->lenth + num2->lenth;
	if (result->data[result->lenth - 1] == 0)
	{
		result->lenth--;
	}
	if (result->sign == 1) // 负数取补码
	{
		compleMent(result); // 取补码
	}
	if (num1->sign == 1) // 负数取补码
		compleMent(num1);
	if (num2->sign == 1) // 负数取补码
		compleMent(num2);
}

void bignum_div(BigNum *result, BigNum *num1, BigNum *num2) {
	
}

int main()
{
	// 定义大数字符串
	char num1[1024] = {0};
	char num2[1024] = {0};
	char result[2048] = {0}; // 结果字符串
	scanf("%s %s", num1, num2); // 输入两个大数

	// 初始化大数
	BigNum *bigNum1 = bignum_create(num1, sizeof(num1));
	BigNum *bigNum2 = bignum_create(num2, sizeof(num2));
	BigNum *bigNum3 = bignum_create(result, sizeof(result));

	printf("num1: %s, num2: %s\n", bignum2char(bigNum1), bignum2char(bigNum2));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	// 测试大数加法
	bignum_add(bigNum3, bigNum1, bigNum2);
	printf("加法结果: %s + %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	memset(result, 0, sizeof(result)); // 清空结果
	bignum_sub(bigNum3, bigNum1, bigNum2); // 测试大数减法
	printf("减法结果: %s - %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	memset(result, 0, sizeof(result)); // 清空结果
	bignum_mul(bigNum3, bigNum1, bigNum2); // 测试大数乘法
	printf("乘法结果: %s * %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	// 释放内存
	free(bigNum1);
	free(bigNum2);
	free(bigNum3);
	return 0;
}
