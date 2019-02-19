#!/bin/bash

rm -r data/tshark_csv
mkdir data/tshark_csv

cd data/pcaps
for pcap in *.pcap; do
	echo "extracting $pcap";
  target=`echo ${pcap} | sed 's/(\([0-9]\))//g'`
	tshark -r $pcap  -T fields -e frame.number -e frame.time -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -E header=y -E separator=, -E quote=d -E occurrence=f > ../tshark_csv/${target}_tshark.csv
done
