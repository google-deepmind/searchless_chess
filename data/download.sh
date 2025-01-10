#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


set -ex

wget https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv

mkdir test
cd test
wget https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag
wget https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag
wget https://storage.googleapis.com/searchless_chess/data/test/state_value_data.bag
cd ..

mkdir train
cd train
for idx in $(seq -f "%05g" 0 2147)
do
  wget https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag
done
wget https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag
wget https://storage.googleapis.com/searchless_chess/data/train/state_value_data.bag
cd ..
