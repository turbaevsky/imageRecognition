#!/bin/bash

cd /home/tur/rasp/camera

for sub in *
do
    cd "/home/tur/rasp/camera/$sub/images/"
    for f in *.jpg
    do
	echo "processing $f file..."
	feh -x $f
	read -rsn1 -p "press a key" a
	if [  $a = "1" ]
	then 
	    echo "/home/tur/rasp/camera/$sub/images/$f,1">>/home/tur/imageCat.csv
	elif [ $a = "2" ]
	then 
		echo "/home/tur/rasp/camera/$sub/images/$f,0">>/home/tur/imageCat.csv
	elif [ $a = "3" ]
	then 
		echo "/home/tur/rasp/camera/$sub/images/$f,2">>/home/tur/imageCat.csv
	elif [ $a = "4" ]
	then
	    echo "/home/tur/rasp/camera/$sub/images/$f,3">>/home/tur/imageCat.csv
	fi
    done
done
