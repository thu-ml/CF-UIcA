#!/bin/sh
#

DOWN=wget
#DOWN=curl

#$DOWN http://files.grouplens.org/datasets/movielens/ml-100k.zip -P MovieLens/zipfiles/
$DOWN http://files.grouplens.org/datasets/movielens/ml-1m.zip -P MovieLens/zipfiles/
#$DOWN http://files.grouplens.org/datasets/movielens/ml-10m.zip -P MovieLens/zipfiles/
#$DOWN http://files.grouplens.org/datasets/movielens/ml-20m.zip -P MovieLens/zipfiles/
