FROM alpine

RUN apk update && apk add tor proxychains-ng bash
RUN apk add procps
RUN mkdir -p /etc/tor/

RUN mkdir -p /root/.tor/hidden
RUN chmod 700 /root/.tor/hidden

# Copy all items
COPY torrc /etc/tor/
RUN tor & sleep 20 && pkill tor

COPY tor.sh /root/

## Entry points
# To run the container, docker build <directory>, docker run -it <image_id>

#ENTRYPOINT ["/root/tor.sh"] #CMD ["tor"]
CMD ["/root/tor.sh"]
