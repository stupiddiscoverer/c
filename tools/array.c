#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct
{
    int size;
    int lenth;
    int sizeOfData;
    char *data;
    // void *get;
    // void *set;
    // void *remove;
    // void *clear;
    // void *add;
    // void *insert;
    // void *indexOf;
    // void *lastIndexOf;
    // void *reverse;
    // void *sort;
    // void *copy;
    // void *toString;
    // void *equals;
    // void *free;
} Array;

void *array_get(Array* array, int index) {
    if (index < 0 || index >= array->lenth) {
        return NULL;
    }
    return array->data + index * array->sizeOfData;
}
void array_set(Array* array, int index, void* value) {
    if (index < 0 || index >= array->lenth) {
        return;
    }
    memcpy(array->data + index * array->sizeOfData, value, array->sizeOfData);
}
void array_remove(Array* array, int index) {
    if (index < 0 || index >= array->lenth) {
        return;
    }
    memmove(array->data + index * array->sizeOfData, array->data + (index + 1) * array->sizeOfData, (array->lenth - 1 - index) * array->sizeOfData);
    array->lenth--;
}
void array_clear(Array* array) {
    if (array->lenth == 0) {
        return;
    }
    memset(array->data, 0, array->size * array->sizeOfData);
    array->lenth = 0;
}
void array_add(Array* array, void* value) {
    if (array->lenth >= array->size) {
        array->size *= 2;
        printf("%p\n", array->data);
        array->data = (char*)realloc(array->data, array->size * array->sizeOfData);
        printf("%p\n", array->data);
    }
    memcpy(array->data + array->lenth * array->sizeOfData, value, array->sizeOfData);
    array->lenth++;
}
void array_insert(Array* array, int index, void* value) {
    if (index < 0 || index > array->lenth) {
        return;
    }
    if (array->lenth >= array->size) {
        array->size *= 2;
        array->data = (char*)realloc(array->data, array->size * array->sizeOfData);
    }
    memmove(array->data + (index + 1) * array->sizeOfData, array->data + index * array->sizeOfData, (array->lenth - index) * array->sizeOfData);
    memcpy(array->data + index * array->sizeOfData, value, array->sizeOfData);
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
    char *str = (char *)malloc(array->lenth * (array->sizeOfData + 1));
    char *p = str;
    for (int i = 0; i < array->lenth; i++) {
        memcpy(p, array->data + i * array->sizeOfData, array->sizeOfData);
        p += array->sizeOfData;
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

Array *array_init(int sizeOfData)
{
    if (sizeOfData <= 0)
    {
        sizeOfData = sizeof(char);
    }

    Array *array = (Array *)malloc(sizeof(Array));
    array->size = 8;
    array->lenth = 0;
    array->sizeOfData = sizeOfData;
    array->data = (char *)malloc(8 * sizeOfData);
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
