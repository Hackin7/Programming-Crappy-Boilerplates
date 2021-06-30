// Inspired by sololearn tutorials
#include <stdio.h>
#include <stdlib.h>

int main(){
    // Allocation of Memory
    int* ptr;
    ptr = malloc(sizeof(int)); // allocate enough size for an int(usually 4 bytes)
    ptr = calloc(2, sizeof(int)); // Memory of 2 ints
    if (ptr == NULL){ // Check if Memory Allocated
        return 0;
    }
    *ptr = 1; *(ptr+1) = 2;

    // Reallocation of memory
    ptr = realloc(ptr, 10 * sizeof(*ptr)); // Should be checking for NILL here but whatever
    *(ptr+2) = 3;
    printf("%d %d %d\n", *ptr, *(ptr+1), *(ptr+2) );

    // Free memory
    free(ptr);
}
