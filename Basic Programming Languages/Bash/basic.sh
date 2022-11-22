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
echo ${command^} ${command^^} # Capitalisation https://linuxhint.com/bash_lowercase_uppercase_strings/
### Conditionals ################################################
echo "### Conditionals #########################################"
VAR=""
if [[ -z  "$VAR" ]]; then
  echo "String is empty"
elif [[ -n  "$VAR" ]]; then
  echo "String is not empty"
else
  echo "All else shouldn't fail but if it does, its here"
fi

if [[ "$VAR" == "value" ]]; then echo "String is value"; fi
if [[ 10 -gt 1 ]]; then echo "greater"; fi
if [[ 0 -lt 1 ]]; then echo "smaller"; fi

# https://stackoverflow.com/questions/4665051/check-if-passed-argument-is-file-or-directory-in-bash
if [ -f "basic.sh" ]; then echo "${PASSED} is a file"; fi

### Arrays ######################################################
# Looping through array
# https://www.cyberciti.biz/faq/unix-linux-bash-script-check-if-variable-is-empty/
echo "### For Loop through array #############################"
arr=("1" "2" "3" "a")
len=${#arr[@]}
## Use bash for loop, compare & match with the respective replacement
for (( i=0; i<$len; i++ )); do 
  echo "  ${arr[$i]}"
done

#https://www.folkstalk.com/2022/09/check-if-value-is-in-bash-array-with-code-examples.html
value="1"
if [[ " ${arr[*]} " =~ " ${value} " ]]; then
    # whatever you want to do when array contains value
    echo "Value in arr"
fi

# https://stackoverflow.com/questions/10586153/how-to-split-a-string-into-an-array-in-bash
IFS=',' read -r -a array <<< "4,5,6"
echo "${array[0]}"
echo ${array[*]}

### Loops #######################################################
# While Loop
# https://www.cyberciti.biz/faq/bash-while-loop/
while :
do
  # Input format is '1 2'
	read -p "Enter two numnbers ( - 1 to quit ) : " a b
	if [ $a -eq -1 ]
	then
		break
	fi
	ans=$(( a + b ))
	echo $ans
done
### Functions ###################################################

sample_function(){
    echo "You called a function where the first parameter was " $1
    return 2 # Can only be numerical values
}
sample_function 1
echo 'Return value is '$?
