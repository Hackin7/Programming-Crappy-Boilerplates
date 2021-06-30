#include <stdio.h>

int say_hello(int a, int b){
    printf("say_hello has been called\n");
}

int say_no(int a, int b){
   printf("no\n");
}
void wants_callback_func(int (*callback)(int,int) ){
    printf("Calling callback function\n    ");
    callback(0,1);
}

int main(){
    // Pointers
    int a = 0;
    int* ptr = &a; // Get pointer
    *ptr = 1;
    int def = *ptr; //Dereference
    printf("Integer Pointers: %d %d %d\n", a, ptr, 0);

    // Pointers and Arrays
    int arr[] = {0,1,2,3,4};
    int *ptr_arr = arr;
    printf("Array Pointers: %d %d\n", *(ptr_arr), *(ptr_arr+2));

    // Function Pointers
    int (*ptr_func)(int, int) = say_hello;
    ptr_func(1,2);
    wants_callback_func(ptr_func);
    int (*ptr_funcs[])(int, int) = {
        say_hello, say_no
    };
    ptr_funcs[1](1,2);

    // void pointers: can be any pointers
    void* ptr_void = &a;
    printf("Void Pointers: %d\n", *(int*)ptr_void);
}
/*
Other things
1. To pass in/out array into function, can pass in a pointer
*/
