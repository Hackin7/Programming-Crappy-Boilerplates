FROM kalilinux/kali-rolling

# Copy all items
WORKDIR /usr/src/app
COPY . .

# Get list of items
RUN apt update
RUN apt install -y python3 python3-pip python

# https://www.gungorbudak.com/blog/2018/08/02/correct-installation-and-configuration-of-pip2-and-pip3/
RUN apt install -y wget
RUN cd /tmp && wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN python2 /tmp/get-pip.py
RUN pip3 install uncompyle6
RUN pip2 install uncompyle6

#Expose ports if needed
#EXPOSE 80 5000
#  docker run -dp 3000:3000 getting-started

ENTRYPOINT  ["bash", "-c", "bash"]
CMD ["bash -c bash"]
