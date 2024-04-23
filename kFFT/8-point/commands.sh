#!/usr/bin/env bash

set -e

cslc ./layout3.csl --fabric-dims=11,3 --fabric-offsets=4,1 \
--params=width:4 --colors=LAUNCH:8 --memcpy --channels=1 -o out-fft
cs_python run.py --name out-fft
