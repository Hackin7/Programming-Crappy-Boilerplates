#!/bin/sh

OVPN_DATA="ovpn-data-example"
HOSTNAME="0.tcp.ngrok.io:18217" #"localhost:1194"
PROTOCOL="tcp" #"udp"
CLIENTNAME="client"

#docker volume rm $OVPN_DATA
docker volume create --name $OVPN_DATA
docker run -v $OVPN_DATA:/etc/openvpn --rm kylemanna/openvpn ovpn_genconfig -u $PROTOCOL://$HOSTNAME
docker run -v $OVPN_DATA:/etc/openvpn --rm -it kylemanna/openvpn ovpn_initpki

docker run -v $OVPN_DATA:/etc/openvpn --rm -it kylemanna/openvpn easyrsa build-client-full $CLIENTNAME nopass

echo "Can do any additional config in the bash shell. Else just exit"
# To allow access of server network devices run
#echo 'push "route 192.168.1.92 255.255.255.0"' >> /etc/openvpn/openvpn.conf
docker run -v $OVPN_DATA:/etc/openvpn --network host -it kylemanna/openvpn bash
