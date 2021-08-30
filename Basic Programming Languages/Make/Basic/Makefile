# Tutorial followed: https://makefiletutorial.com/

# The main target is always the first
function: hello target_1 var_test var_test_2 print test_binary
	echo "### Makefile Tutorial ########################"

### Sample Format #########################################
# targets : prerequisites
# <tab> command_1
# <tab> command_2

hello:
	echo "hello world" # Have to put a tab in front of the command

### Prerequisites #########################################

# Multiple targets
target_1: target_2 target_3
	echo "target_1"
	
target_2: target_3
	echo "target_2"
	
target_3:
	echo "target_3"
	
### Variables #############################################
var1 = abc def

var_test: $(var1)
	echo "### Variables #########################################"
	echo Target Name: $@
	echo $(var1)
	
abc:
	echo "abc"
	
def:
	echo "def"
	
# Recursive variable. This will print "later" below
one = one ${later_variable}
# Simply expanded variable. This will not print "later" below
two := two ${later_variable}
later_variable = later
var_test_2: 
	echo $(one)
	echo $(two)

### %/* Wildcard ##########################################
	
print: $(wildcard *.c)
	ls -la  $?

### Actual Compilation ####################################
CC := gcc
test_binary: test.c
