#include <stdio.h>

struct test1{
    int x;
    int y;
};

typedef struct{
    int x;
    int y;
} test2;

union u{
    int x;
    int y; //Try chaning var type
};

int main(){
    struct test1 a = {1, 2};
    test2 b;
    union u c;

    a.x = 3;
    printf("a: %d %d \n",a.x, a.y);

    c.x = 2;
    printf("c: %d %d \n",c.x, c.y);

    struct test1 *ptr= &a;
    ptr->y = 1;
    printf("a: %d %d \n",ptr->x, ptr->y);
}

