# I deployed this at online-decompiler.herokuapp.com

FROM kalilinux/kali-rolling

### Install Basic Tools ####################################
RUN apt update
# Development tools
RUN apt-get install -y build-essential cmake git openssl libssl-dev python3 \
    autoconf automake libtool pkg-config m4 zlib1g-dev upx doxygen graphviz tar\
    wget

WORKDIR /tmp

### Retdec Decompiler #####################################
# https://github.com/avast/retdec
RUN wget https://github.com/avast/retdec/releases/download/v4.0/retdec-v4.0-debian-64b.tar.xz
RUN tar -xvf retdec-v4.0-debian-64b.tar.xz -C ~

### Ghidra Decompiler #####################################
WORKDIR /tmp
RUN apt install -y unzip default-jdk
RUN wget https://ghidra-sre.org/ghidra_9.2.3_PUBLIC_20210325.zip
RUN unzip ghidra_9.2.3_PUBLIC_20210325.zip
RUN mv ghidra_9.2.3_PUBLIC /root/ghidra


### Snowman Decompiler ####################################
# https://github.com/yegord/snowman/blob/master/doc/build.asciidoc
WORKDIR /root
RUN git clone https://github.com/yegord/snowman.git
#RUN cd ~/snowman && cmake --cmake --build . --target install

# to decompile, run nocode <file>

RUN rm -rf /tmp/*

### App Files ############################################
# Copy all items
WORKDIR /root
RUN apt -y install zip python3-pip
RUN pip3 install flask
COPY . .
RUN ls -a
EXPOSE 8000

### Entry points #########################################
# To run the container, docker build <directory>, docker run -it <image_id>
WORKDIR /root
ENTRYPOINT ["python3", "/root/main.py"]
CMD ["python3", "/root/main.py"]

#ENTRYPOINT  ["bash", "-c", "bash"]
#CMD ["bash -c bash"]
