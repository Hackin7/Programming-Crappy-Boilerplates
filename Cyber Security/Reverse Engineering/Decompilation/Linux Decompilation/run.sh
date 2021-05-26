#!/bin/sh
# name:tag
sudo docker build . -t test:test && \
sudo docker run \
    -p 8000:8000 \
    -v "$(pwd)":/mnt \
    -it \
    test:test
