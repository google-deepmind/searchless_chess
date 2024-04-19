#!/bin/bash
set -ex

download_if_not_exist() {
	url=$1
	file=$(basename "$url")

	if [ ! -f "$file" ]; then
		wget "$url"
	else
		echo "$file already exists, skipping download."
	fi
}

download_if_not_exist https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/puzzles.csv

mkdir -p test
cd test
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/test/state_value_data.bag
cd ..

mkdir -p train
cd train
for idx in $(seq -f "%05g" 0 2147)
do
	download_if_not_exist https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag
done
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag
download_if_not_exist https://storage.googleapis.com/searchless_chess/data/train/state_value_data.bag
cd ..


