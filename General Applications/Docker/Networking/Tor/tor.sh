#!/bin/bash

# sh

# https://ryanstutorials.net/bash-scripting-tutorial/bash-loops.php
all_ports() {
    for value in {2..65535}
    do
        echo 'HiddenServicePort '$value '127.0.0.1:'$value >> /etc/tor/torrc
    done
}

all_ports
cat /root/.tor/hidden/hostname 
tor
cat /root/.tor/hidden/hostname
