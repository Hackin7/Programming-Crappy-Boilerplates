#include <stdio.h>
// https://www.educative.io/edpresso/c-reading-data-from-a-file-using-fread
void main() {
    char buffer[20]; // Buffer to store data
    FILE * stream;
    stream = fopen("file.txt", "r");
    int count = fread(&buffer,20, 2* sizeof(char), stream);
   
    fclose(stream);
    
    // Printing data to check validity
    printf("%c %c\n", buffer[0], buffer[1]);
    printf("Data read from file: %s \n", buffer);
    printf("Elements read: %d\n", count);
    
}
