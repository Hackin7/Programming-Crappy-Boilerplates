#!/bin/sh

OVPN_DATA="ovpn-data-example"
#HOSTNAME="localhost:1194"
PROTOCOL="tcp" #"udp"
CLIENTNAME="client"

docker run -v $OVPN_DATA:/etc/openvpn -d -p 1194:1194/$PROTOCOL --cap-add=NET_ADMIN --network host kylemanna/openvpn
#docker run -v $OVPN_DATA:/etc/openvpn --rm -it kylemanna/openvpn easyrsa build-client-full $CLIENTNAME nopass
docker run -v $OVPN_DATA:/etc/openvpn --rm kylemanna/openvpn ovpn_getclient $CLIENTNAME > $CLIENTNAME.ovpn
