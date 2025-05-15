#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/array.h"



void *array_get(Array* array, int index) {
    if (index < 0 || index >= array->lenth) {
        return NULL;
    }
    return array->data + index * array->sigleDataSize;
}
void array_set(Array* array, int index, void* value) {
    if (index < 0 || index >= array->lenth) {
        return;
    }
    memcpy(array->data + index * array->sigleDataSize, value, array->sigleDataSize);
}
void array_remove(Array* array, int index) {
    if (index < 0 || index >= array->lenth) {
        return;
    }
    memmove(array->data + index * array->sigleDataSize, array->data + (index + 1) * array->sigleDataSize, (array->lenth - 1 - index) * array->sigleDataSize);
    array->lenth--;
}
void array_clear(Array* array) {
    if (array->lenth == 0) {
        return;
    }
    memset(array->data, 0, array->size * array->sigleDataSize);
    array->lenth = 0;
}
void array_add(Array* array, void* value) {
    if (array->lenth >= array->size) {
        array->size *= 2;
        printf("%p\n", array->data);
        array->data = (char*)realloc(array->data, array->size * array->sigleDataSize);
        printf("%p\n", array->data);
    }
    memcpy(array->data + array->lenth * array->sigleDataSize, value, array->sigleDataSize);
    array->lenth++;
}
void array_insert(Array* array, int index, void* value) {
    if (index < 0 || index > array->lenth) {
        return;
    }
    if (array->lenth >= array->size) {
        array->size *= 2;
        array->data = (char*)realloc(array->data, array->size * array->sigleDataSize);
    }
    memmove(array->data + (index + 1) * array->sigleDataSize, array->data + index * array->sigleDataSize, (array->lenth - index) * array->sigleDataSize);
    memcpy(array->data + index * array->sigleDataSize, value, array->sigleDataSize);
    array->lenth++;
}
void *array_indexOf();
void *array_lastIndexOf();
void *array_reverse();
void *array_sort();
void *array_copy();
void *array_toString(Array* array) {
    if (array->lenth == 0) {
        return NULL;
    }
    char *str = (char *)malloc(array->lenth * (array->sigleDataSize + 1));
    char *p = str;
    for (int i = 0; i < array->lenth; i++) {
        memcpy(p, array->data + i * array->sigleDataSize, array->sigleDataSize);
        p += array->sigleDataSize;
        *p++ = ' ';
    }
    *p = '\0';
    return str;
}
void *equals();
void array_free(Array* array) {
    if (array == NULL) {
        return;
    }
    free(array->data);
    free(array);
}

Array *array_init(int sigleDataSize)
{
    if (sigleDataSize <= 0)
    {
        sigleDataSize = sizeof(char);
    }

    Array *array = (Array *)malloc(sizeof(Array));
    array->size = 8;
    array->lenth = 0;
    array->sigleDataSize = sigleDataSize;
    array->data = (char *)malloc(8 * sigleDataSize);
    return array;
}

int main(int argc, char const *argv[])
{
    Array *array = array_init(sizeof(int));
    int a = 1;
    int b = 2;
    int c = 3;
    array_add(array, &a);
    array_add(array, &b);
    array_add(array, &c);
    int *p = (int *)array_get(array, 0);
    printf("Array[0]: %d\n", *p);
    
    printf("Array: %s\n", (char *)array_toString(array));
    return 0;
}
