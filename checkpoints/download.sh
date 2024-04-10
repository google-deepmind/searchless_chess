#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
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

wget https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip
wget https://storage.googleapis.com/searchless_chess/checkpoints/136M.zip
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip

unzip 9M.zip
unzip 136M.zip
unzip 270M.zip

rm 9M.zip
rm 136M.zip
rm 270M.zip
