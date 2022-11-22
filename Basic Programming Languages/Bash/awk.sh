#!/bin/bash

# https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
cat <<EOF > print.csv
header1,header2,header3
1,2,3
4,5,a
EOF

awk '
# Data may have trailing unwanted characters, this function removes them
# https://stackoverflow.com/questions/20600982/trim-leading-and-trailing-spaces-from-a-string-in-awk
function trimSpace(){ 
  for (i=1; i<=NF; i++){ # NF - Number of fields
    gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", $i);
    gsub(/\n/, "", $i);
  }
}
function test(string){
  print "test: "string;
}
BEGIN {
  ### Handle Fields ##################################################
  FS=","; # FS - Field Separator
  getline; 
  NR=1;  # NR - Line number of current record
  print "Variable Options:"
  for (i=1; i<=NF; i++){ # NF - Number of fields
    print "$"i") "$i"";
  }
  
  ### Variable & Functions ###########################################
  variable = 0
  variable = variable + 1
  print "Variable: " variable;
  test(variable)
  
  ### String formatting ##############################################
  # https://www.cyberciti.biz/faq/awk-dont-print-newline-on-linux-unix-macos/
  printf "%s | %5s | %-5c| %c \n", "a", "b", "c", "de";
  printf " %f | %5f | %5.2f | %d \n", 1.0, 1.1, 1.2, 2;
  
  print "### Table ##################";
}
NR > 1{
  if ($1 == "1" && $2 == "2"){
    print $0","$1,$2,$3
    printf ""
  }else{
    print "Irrelevant";
  }
}
END{
  print "### END ####################";
}
' print.csv

rm print.csv

