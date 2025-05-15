typedef struct
{
    int size;
    int lenth;
    int sigleDataSize;
    char *data;
} Array;

Array* array_init(int sigleDataSize);
void *array_get(Array* array, int index);
void array_set(Array* array, int index, void* value);
void array_remove(Array* array, int index);
void array_clear(Array* array);
void array_add(Array* array, void* value);
void array_insert(Array* array, int index, void* value);
void *array_indexOf();
void *array_lastIndexOf();
void *array_reverse();
void *array_sort();
void *array_copy();
void *array_toString(Array* array);
