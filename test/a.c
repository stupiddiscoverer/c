#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    int nearIndex;
    struct node* next;
} Node;

Node* buildList(int** roads, int roadsSize)
{
    int *arr = (int*)roads;
    // printf("%d, %d, %d\n", *arr, *(arr+1), **roads);
    Node *list = (Node*) malloc(sizeof(Node) * (roadsSize + 1));
    Node **endPtr = (Node**) malloc(sizeof(Node*) * (roadsSize + 1));
    memset(list, 0, sizeof(Node) * (roadsSize + 1));

    for (int i=0; i<roadsSize+1; i++)
    {
        endPtr[i] = &list[i];
        list[i].nearIndex = -1;
    }

    for (int i=0; i<roadsSize; i++)
    {
        if ((endPtr[arr[i*2]])->nearIndex != -1)
        {
            Node *temp = (Node*) malloc(sizeof(Node));
            temp->next = 0;
            (endPtr[arr[i*2]])->next = temp;
            endPtr[arr[i*2]] = temp;
        }
        (endPtr[arr[i*2]])->nearIndex = arr[i*2+1];
        
        if ((endPtr[arr[i*2+1]])->nearIndex != -1)
        {
            Node *temp = (Node*) malloc(sizeof(Node));
            temp->next = 0;
            (endPtr[arr[i*2+1]])->next = temp;
            endPtr[arr[i*2+1]] = temp;
        }
        (endPtr[arr[i*2+1]])->nearIndex = arr[i*2];
    }
    free(endPtr);
    return list;
}

void printList(Node *list, int n)
{
    Node **endPtr = (Node**) malloc(sizeof(void*) * n);
    for(int i=0; i<n; i++)
    {
        endPtr[i] = &list[i];
    }
    int empty = 0;
    while (empty == 0)
    {
        empty = 1;
        for(int i=0; i<n; i++)
        {
            if (endPtr[i])
            {
                printf("%d\t", endPtr[i]->nearIndex);
                endPtr[i] = endPtr[i]->next;
                empty = 0;
            }
            else
            {
                printf(" \t");
            }
        }
        puts("");
    }
}

void gotoCapital(long long *fuelCost, Node *list, int myIndex, int destinationIndex, int *peopleCount, int seats)
{
    Node *temp = &list[myIndex];
    while (temp->next || temp->nearIndex != destinationIndex)
    {
        if (temp->nearIndex != destinationIndex)
        {
            gotoCapital(fuelCost, list, temp->nearIndex, myIndex, peopleCount, seats);
        }
        if (temp->next)
        {
            Node* next = temp->next;
            temp->nearIndex = next->nearIndex;
            temp->next = next->next;
            free(next);
        }
        else
        {
            temp->nearIndex = destinationIndex;
        }
    }
    if (myIndex != destinationIndex)
    {
        peopleCount[destinationIndex] += peopleCount[myIndex];
        (*fuelCost) += (peopleCount[myIndex] - 1) / seats + 1;
    }
}

long long minimumFuelCost(int** roads, int roadsSize, int* roadsColSize, int seats) {
    if (roadsSize == 0)
    {
        return 0;
    }
    Node *list = buildList(roads, roadsSize);
    int *peopleCount = malloc((roadsSize+1)*sizeof(int));
    //初始化每个城市的人数
    for(int i=1; i<roadsSize+1; i++)
    {
        peopleCount[i] = 1;
    }
    peopleCount[0] = 0;
    long long fuelCost = 0;
    gotoCapital(&fuelCost, list, 0, 0, peopleCount, seats);
    free(list);
    free(peopleCount);
    
    return (long long)fuelCost;
}

int main(int argc, char const *argv[])
{
    // double b = (double)0.1234567;
    // float a = 0.23456789;
    // printf("%.10f, %.10f\n", a, b*12.345);
    printf("sizeof(void*) = %lu\n", sizeof(void*));
    int a[][2] = {{3, 1},{3,2},{1,0},{0,4},{0,5},{4,6}};
    // int a[][2] = {{0,1},{0,2},{0,3}};
    // int a[][2] = {};
    printf("%lu, %lu\n", sizeof(Node*), sizeof(a) / sizeof(int) / 2);
    long long fuelcost = minimumFuelCost((int**)a, sizeof(a) / sizeof(int) / 2, 0, 2);
    printf("%lld\n", fuelcost);
    return 0;
}
