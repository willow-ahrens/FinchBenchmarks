#include <stdint.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv){
    uint8_t r = 255;
    uint8_t s = 128;
    for(int i = 0; i < 20000000; i++){
        r = lrintf(r*0.5 + s*0.5);
    }
    printf("%d",r);
    return 0;
}
