#!/bin/bash
cd data/pcaps
rm -r ../csv
mkdir ../csv

for pcap in *.pcap; do
        target=`echo ${pcap} | sed 's/(\([0-9]\))//g'`
        echo "extracting Wireshark columns for $pcap";
        echo 'Source;;Destination;;Protocol;;Length;;Info' > ../csv/${target}_extracted.csv
        tshark -r $pcap  -o 'gui.column.format:"Source","%s","c","%Cus:frame.encap_type","Destination","%d","c","%Cus:frame.encap_type","Protocol","%p","c","%Cus:frame.encap_type","Length","%L","c","%Cus:frame.encap_type","Info","%i"' | sed 's/ Ethernet /;;/g' >> ../csv/${target}_extracted.csv
done

