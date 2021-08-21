#!/bin/bash

sample_function(){
    echo "You called a function where the first parameter was " $1
    return '2'
}
sample_function 1
echo 'Return value is '$?
