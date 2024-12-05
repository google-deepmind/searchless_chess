# Amortized Planning with Large-Scale Transformers: A Case Study on Chess

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/searchless_chess/master/overview.svg" alt="Overview figure"/>
</p>


This repository provides an implementation of our NeurIPS 2024 paper [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494).

> This paper uses chess, a landmark planning problem in AI, to assess transformers’ performance on a planning task where memorization is futile — even at a large scale.
To this end, we release ChessBench, a large-scale benchmark dataset of 10 million chess games with legal move and value annotations (15 billion data points) provided by Stockfish 16, the state-of-the-art chess engine.
We train transformers with up to 270 million parameters on ChessBench via supervised learning and perform extensive ablations to assess the impact of dataset size, model size, architecture type, and different prediction targets (state-values, action-values, and behavioral cloning).
Our largest models learn to predict action-values for novel boards quite accurately, implying highly non-trivial generalization.
Despite performing no explicit search, our resulting chess policy solves challenging chess puzzles and achieves a surprisingly strong Lichess blitz Elo of 2895 against humans (grandmaster level).
We also compare to Leela Chess Zero and AlphaZero (trained without supervision via self-play) with and without search.
We show that, although a remarkably good approximation of Stockfish’s search-based algorithm can be distilled into large-scale transformers via supervised learning, perfect distillation is still beyond reach, thus making ChessBench well-suited for future research.


## Contents

```
.
|
├── BayesElo                        - Elo computation (need to be installed)
|
├── checkpoints                     - Model checkpoints (need to be downloaded)
|   ├── 136M
|   ├── 270M
|   └── 9M
|
├── data                            - Datasets (need to be downloaded)
|   ├── eco_openings.csv
|   ├── test
|   ├── train
|   └── puzzles.csv
|
├── lc0                             - Leela Chess Zero (needs to be installed)
|
├── src
|   ├── engines
|   |   ├── constants.py            - Engine constants
|   |   ├── engine.py               - Engine interface
|   |   ├── lc0_engine.py           - Leela Chess Zero engine
|   |   ├── neural_engines.py       - Neural engines
|   |   └── stockfish_engine.py     - Stockfish engine
|   |
|   ├── bagz.py                     - Readers for our .bag data files
|   ├── config.py                   - Experiment configurations
|   ├── constants.py                - Constants, interfaces, and types
|   ├── data_loader.py              - Data loader
|   ├── metrics_evaluator.py        - Metrics (e.g., Kendall's tau) evaluator
|   ├── puzzles.py                  - Puzzle evaluation script
|   ├── searchless_chess.ipynb      - Model analysis notebook
|   ├── tokenizer.py                - Chess board tokenization
|   ├── tournament.py               - Elo tournament script
|   ├── train.py                    - Example training + evaluation script
|   ├── training.py                 - Training loop
|   ├── training_utils.py           - Training utility functions
|   ├── transformer.py              - Decoder-only Transformer
|   └── utils.py                    - Utility functions
|
├── Stockfish                       - Stockfish (needs to be installed)
|
├── README.md
└── requirements.txt                - Dependencies
```


## Installation

Clone the source code into a local directory:

```bash
git clone https://github.com/google-deepmind/searchless_chess.git
cd searchless_chess
```

This repository requires Python 3.10.
`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).
Then, create and activate the conda environment:

```bash
conda create --name searchless_chess python=3.10
conda activate searchless_chess
```

Install `pip` and use it to install all the dependencies:

```bash
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support.

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use (CUDA 12 in the example above).
Please see the [JAX documentation](https://github.com/jax-ml/jax#installation) for more details.

### Installing Stockfish

Download and compile the latest version of Stockfish (for Unix-like systems):

```bash
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```


### Installing Leela Chess Zero

Follow the [Lc0 download instructions](https://github.com/LeelaChessZero/lc0?tab=readme-ov-file#downloading-source), i.e.,

```bash
git clone -b release/0.30 --recurse-submodules https://github.com/LeelaChessZero/lc0.git
```

Then build the engine as described in the [Lc0 build instructions](https://github.com/LeelaChessZero/lc0?tab=readme-ov-file#building-and-running-lc0).

We evaluate Lc0 with the largest-possible network from [Lc0's model catalogue](https://lczero.org/play/networks/bestnets/), i.e., the `Large` network.
To download that network, run the following command:

```bash
cd lc0/build/release
wget https://storage.lczero.org/files/768x15x24h-t82-swa-7464000.pb.gz
gzip -d 768x15x24h-t82-swa-7464000.pb.gz
cd ../../..
```

### Installing BayesElo

To compute the Elos for the different agents, we require [BayesElo](https://www.remi-coulom.fr/Bayesian-Elo/), which can be installed as follows:

```bash
wget https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2
tar -xvjf bayeselo.tar.bz2
cd BayesElo
make bayeselo
cd ..
```


### Downloading the Datasets

To download our datasets to the correct locations, run the following command:

```bash
cd data
./download.sh
cd ..
```

We also provide the individual dataset download links in the following table
(the action-value dataset is sharded into 2148 files due to its size and only
the link to the first shard is listed below):

| Split | Action-Value | Behavioral Cloning | State-Value | Puzzles |
|------ | ------------ | ------------------ | ----------- | ------- |
| Train | [1.2 GB](https://storage.googleapis.com/searchless_chess/data/train/action_value-00000-of-02148_data.bag) (of 1.1 TB) | [34 GB](https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag) | [36 GB](https://storage.googleapis.com/searchless_chess/data/train/state_value_data.bag) | - |
| Test  | [141 MB](https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag) | [4.1 MB](https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag) | [4.4 MB](https://storage.googleapis.com/searchless_chess/data/test/state_value_data.bag) | [4.5 MB](https://storage.googleapis.com/searchless_chess/data/puzzles.csv) |


### Downloading the Model Checkpoints

To download the pretrained models to the correct locations, run the following command:

```bash
cd checkpoints
./download.sh
cd ..
```


## Usage

Before running any code, make sure to activate the conda environment and set the `PYTHONPATH`:

```bash
conda activate searchless_chess
export PYTHONPATH=$(pwd)/..
```

### Training

To train a model locally, run the following command:

```bash
cd src
python train.py
cd ..
```
The model checkpoints will be saved to `/checkpoints/local`.

### Puzzles

To evaluate a model's puzzle accuracy, run the following command:

```bash
cd src
python puzzles.py --num_puzzles 10 --agent=local
cd ..
```

`puzzles.py` supports the following agents:

* the locally trained model: `local`
* the pretrained models: `9M`, `136M`, and `270M`
* the Stockfish engines: `stockfish` and `stockfish_all_moves`
* the Lc0 engines: `leela_chess_zero_depth_1`, `leela_chess_zero_policy_net`, and `leela_chess_zero_400_sims`


### Tournament Elo

To compute the Elo for the different agents, run the tournament to play games between them and then compute the Elo for the PGN file generated by the tournament (more information on BayesElo can be found [here](https://www.remi-coulom.fr/Bayesian-Elo/)):

```bash
cd src
python tournament.py --num_games=200

cd ../BayesElo

./bayeselo
> ...
ResultSet>readpgn ../data/tournament.pgn
> N game(s) loaded, 0 game(s) with unknown result ignored.
ResultSet>elo
ResultSet-EloRating>mm
> 00:00:00,00
ResultSet-EloRating>exactdist
> 00:00:00,00
ResultSet-EloRating>ratings
> ...

cd ..
```

### Analysis Notebook

To investigate the model's behavior (e.g., to compute the win percentage for all legal moves), start a notebook server and then open `src/searchless_chess.ipynb` in your browser:

```bash
jupyter notebook
```


## Citing this work

```latex
@inproceedings{ruoss2024amortized,
  author       = {Anian Ruoss and
                  Gr{\'{e}}goire Del{\'{e}}tang and
                  Sourabh Medapati and
                  Jordi Grau{-}Moya and
                  Li Kevin Wenliang and
                  Elliot Catt and
                  John Reid and
                  Cannada A. Lewis and
                  Joel Veness and
                  Tim Genewein},
  title        = {Amortized Planning with Large-Scale Transformers: A Case Study
                  on Chess},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

The model weights are licensed under Creative Commons Attribution 4.0 (CC-BY).
You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Some portions of the dataset are in the public domain by a
Creative Commons CC0 license from lichess.org.
The remainder of the dataset is licensed under
Creative Commons Attribution 4.0 (CC-BY).
You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode.

Unless required by applicable law or agreed to in writing, software and
materials distributed under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
