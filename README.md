# CF-UIcA

This repository contains the implementation for the paper:

**Chao Du, Chongxuan Li, Yin Zheng, Jun Zhu, Bo Zhang. [Collaborative Filtering with User-Item Co-Autoregressive Models.](https://arxiv.org/abs/1612.07146)** In AAAI 2018.

## Datasets

To download and preprocess the Movielens 1M dataset:

```
$ cd data/
$ ./download.sh
$ python preprocess.py
$ python preprocess_implicit.py
```

## Usage

To run rating prediction experiments:

```
$ python CF-UIcA_rating-prediction.py
```

To run top-N recommendation experiments:

```
$ python CF-UIcA_topN-recommendation.py
```

## Reference

**Please cite our AAAI'18 paper if you find it is useful. Thanks!**

    @inproceedings{du2018collaborative,
        title={Collaborative Filtering with User-Item Co-Autoregressive Models},
        author={Chao Du and Chongxuan Li and Yin Zheng and Jun Zhu and Bo Zhang},
        booktitle={AAAI},
        year={2018}
    }
