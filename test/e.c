#include <stdio.h>
#include <string.h>

int count[100000] = {0};

void initialCount(int*a, int *p, int n) {
    memset(count, 0, sizeof(count));
    for (int i = 0; i < n; i++)
    {
        if (p[i] > count[a[i]])
        {
            count[a[i]] = p[i];
        }
    }
    for (int i = 1; i < sizeof(count); i++)
    {
        if (count[i] < count[i-1])
        {
            count[i] = count[i-1];
        }
    }
}

int main(int argc, char const *argv[])
{
    int T;
    scanf("%d", &T);
    int result[T];
    for (int i = 0; i < T; i++)
    {
        int n, m;
        scanf("%d%d", &n, &m);
        int a[n], p[n];
        int b[m];
        for (int j = 0; j < n; j++)
        {
            scanf("%d", &a[j]);
        }
        for (int j = 0; j < n; j++)
        {
            scanf("%d", &p[j]);
        }
        for (int j = 0; j < m; j++)
        {
            scanf("%d", &b[j]);
        }
        result[i] = 0;
        initialCount(a, p, n);
        for (int j = 0; j < m; j++)
        {
            result[i] += count[b[j]];
        }
        //printf("%d\n",result[i]);
    }
    for (int i = 0; i < T; i++){
        printf("%d\n",result[i]);
    }
    return 0;
}