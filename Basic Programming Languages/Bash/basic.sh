#!/bin/bash

### Variables ###################################################
first_name=Good
greeting=Hello whoami # Injection
greeting="Hello World"

# Command substitution happens in a subshell and changes to variables in the
#   subshell will not alter variables in the master shell
#command=`whoami` # Discouraged
command=$(whoami)

echo $first_name $greeting $command

### Conditionals ################################################

### Loops #######################################################

### Functions ###################################################

sample_function(){
    echo "You called a function where the first parameter was " $1
    return '2'
}
sample_function 1
echo 'Return value is '$?
