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
	if (len == 0)
	{
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

void compleMent(BigNum *num)
{
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
	compleMent(num);		// 取补码
}

char *bignum2char(BigNum *num)
{
	if (num->lenth == 0)
	{
		num->data[0] = '0';
		return num->data;
	}
	if (num->sign == 1) // 负数
	{
		compleMent(num); // 取补码
	}
	reverseAndMinus(num->data, num->lenth, -'0'); // 反转
	if (num->sign == 1)							  // 负数
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
		if (num[0] == '0' || num[i] < '0' || num[i] > '9')
		{
			printf("Invalid number format\n");
			free(bignum);
			return NULL;
		}
	}
	reverseAndMinus(num, bignum->lenth, '0');			  // 小端存储，负数取补码
	if (bignum->sign == 1)								  // 负数取补码
		compleMent(bignum);								  // 负数取补码
	memset(num + bignum->lenth, 0, size - bignum->lenth); // 清空多余的部分
	return bignum;
}
// 大数加法 num1 = num1+num2 减法就是把被减数sign取反 sign=!sign
void bignum_add(BigNum *result, BigNum *num1, BigNum *num2)
{
	int maxlen = num1->lenth;
	// int sign = num1->sign;
	if (num1->lenth < num2->lenth)
	{
		maxlen = num2->lenth;
		// sign = num2->sign;
	}
	for (int i = num1->lenth; i <= maxlen; i++)
	{
		num1->data[i] = 9 * num1->sign; // 补齐长度
	}
	for (int i = num2->lenth; i <= maxlen; i++)
	{
		num2->data[i] = 9 * num2->sign; // 补齐长度
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
	for (int i = num1->lenth; i <= maxlen; i++)
	{
		num1->data[i] = 0;
	}
	for (int i = num2->lenth; i <= maxlen; i++)
	{
		num2->data[i] = 0;
	}
	while (maxlen >= 0 && result->data[maxlen] == 9 * result->sign)
	{
		result->data[maxlen] = 0; // 清除多余的部分
		maxlen--;
	}
	result->lenth = maxlen + 1;
}

void bignum_sub(BigNum *result, BigNum *num1, BigNum *num2)
{
	changeSign(num2);				// 取反
	bignum_add(result, num1, num2); // 调用加法
	changeSign(num2);				// 取反
}

int bignum_cmp(BigNum *num1, BigNum *num2) // 大于0表示num1>num2，小于0表示num1<num2，等于0表示相等
{
	if (num1->sign != num2->sign) // 符号不同
	{
		return num2->sign - num1->sign;
	}
	if (num1->lenth != num2->lenth) // 长度不同
	{
		return (num1->lenth > num2->lenth) * (1 - 2 * num1->sign);
	}
	for (int i = num1->lenth - 1; i >= 0; i--)
	{
		if (num1->data[i] != num2->data[i])
		{
			return (num1->data[i] - num2->data[i]) * (1 - 2 * num1->sign);
		}
	}
	return 0;
}

void bignum_mul(BigNum *result, BigNum *num1, BigNum *num2)
{
	result->sign = num1->sign ^ num2->sign; // 结果符号
	if (num1->sign == 1)					// 负数取补码
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

void bignum_div(BigNum *result, BigNum *num1, BigNum *num2, BigNum *mod)
{
	if (num2->lenth == 0)
	{
		printf("除数不能为0\n");
		return;
	}
	mod->lenth = num1->lenth;
	memcpy(mod->data, num1->data, num1->lenth); // 余数就是被除数
	mod->sign = num1->sign;						// 余数符号
	if (num1->lenth < num2->lenth)
	{
		result->lenth = 0;
		return;
	}
	if (mod->sign == 1) // 负数取补码
		compleMent(mod);
	if (num2->sign == 1) // 负数取补码
		compleMent(num2);
	result->sign = num1->sign ^ num2->sign; // 结果符号
	result->lenth = num1->lenth - num2->lenth + 1;

	int pre = 0;
	for (int i = 1; i <= result->lenth; i++)
	{
		// 先取前2位计算result->data[len - i], 然后mod = mod - num2 * result->data[len - i]
		// 假如mod < 0 则result->data[len - i]--, mod = mod + num2，因为2位数的除法已经使得result->data[len - i]的误差在1以内了
		int n;
		if (num2->lenth > 1)
		{
			n = (pre * 100 + mod->data[mod->lenth - i] * 10 + mod->data[mod->lenth - i - 1]) / (num2->data[num2->lenth - 1] * 10 + num2->data[num2->lenth - 2]);
			if (n > 0)
			{
				// mod=mod[mod.len-i-num2.len + 1, mod.len-i] - num2 * n
				// 这里是小端存储，所以要从低位开始减
				for (int j = 0; j < num2->lenth; j++)
				{
					int num = n * num2->data[j];
					int tempModIndex = mod->lenth - i - num2->lenth + j + 1;
					mod->data[tempModIndex] -= num % 10;
					mod->data[tempModIndex + 1] -= num / 10;
					if (j == num2->lenth - 1) {
						break;
					}
					if (mod->data[tempModIndex] < 0)
					{
						mod->data[tempModIndex] += 10;
						mod->data[tempModIndex + 1]--;
					}
				}
				if (mod->data[mod->lenth - i] < 0)
				{
					n--;
					for (int j = 0; j < num2->lenth; j++)
					{
						int tempModIndex = mod->lenth - i - num2->lenth + j + 1;
						mod->data[tempModIndex] += num2->data[j];
						if (mod->data[tempModIndex] > 9)
						{
							mod->data[tempModIndex] -= 10;
							mod->data[tempModIndex + 1]++;
						}
					}
				}
			}
		}
		else
		{
			int num = pre * 10 + mod->data[mod->lenth - i];
			n = num / num2->data[0];
			mod->data[mod->lenth - i] = num % num2->data[0];
		}
		result->data[result->lenth - i] = n;
		if (pre > 0)
		{
			mod->data[mod->lenth - i + 1] = 0;
		}
		pre = mod->data[mod->lenth - i];
	}
	for (int i = mod->lenth - 1; i >= 0; i--)
	{
		if (mod->data[i] == 0)
		{
			mod->lenth--;
		}
	}
	if (result->data[result->lenth - 1] == 0)
	{
		result->lenth--;
	}
	if (num2->sign == 1) // 负数取补码
		compleMent(num2);
	if (result->sign == 1)	// 负数取补码
		compleMent(result); // 取补码
	if (mod->sign == 1)		// 负数取补码
		compleMent(mod);	// 取补码
}

int main()
{
	// 定义大数字符串
	char num1[1024] = {0};
	char num2[1024] = {0};
	char result[2048] = {0};	// 结果字符串
	char mod[1024] = {0};		// 余数字符串
	scanf("%s %s", num1, num2); // 输入两个大数

	// 初始化大数
	BigNum *bigNum1 = bignum_create(num1, sizeof(num1));
	if (bigNum1 == NULL)
		return -1;
	BigNum *bigNum2 = bignum_create(num2, sizeof(num2));
	if (bigNum1 == NULL)
		return -1;
	BigNum *bigNum3 = bignum_create(result, sizeof(result));
	BigNum *bigNum4 = bignum_create(mod, sizeof(mod));

	printf("num1: %s, num2: %s\n", bignum2char(bigNum1), bignum2char(bigNum2));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	// 测试大数加法
	bignum_add(bigNum3, bigNum1, bigNum2);
	printf("加法结果: %s + %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	memset(result, 0, sizeof(result));			 // 清空结果
	bignum_sub(bigNum3, bigNum1, bigNum2);		 // 测试大数减法
	printf("减法结果: %s - %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	bigNum1 = bignum_create(num1, sizeof(num1)); // 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2)); // 重新初始化大数
	memset(result, 0, sizeof(result));			 // 清空结果
	bignum_mul(bigNum3, bigNum1, bigNum2);		 // 测试大数乘法
	printf("乘法结果: %s * %s = %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3));
	bigNum1 = bignum_create(num1, sizeof(num1));	// 重新初始化大数
	bigNum2 = bignum_create(num2, sizeof(num2));	// 重新初始化大数
	int big = bignum_cmp(bigNum1, bigNum2); // 测试大数比较
	if (big > 0)
	{
		printf("num1 > num2\n");
	}
	else if (big < 0)
	{
		printf("num1 < num2\n");
	}
	else
	{
		printf("num1 = num2\n");
	}
	memset(result, 0, sizeof(result));				// 清空结果
	bignum_div(bigNum3, bigNum1, bigNum2, bigNum4); // 测试大数除法
	printf("除法结果: %s / %s = %s,余数: %s\n", bignum2char(bigNum1), bignum2char(bigNum2), bignum2char(bigNum3), bignum2char(bigNum4));
	// 释放内存
	free(bigNum1);
	free(bigNum2);
	free(bigNum3);
	return 0;
}
