#!/bin/sh

sudo docker build . -t test:test # name:tag
sudo docker run \
    --network host \
    -v "$(pwd)":/mnt \
    -it \
    test:test
