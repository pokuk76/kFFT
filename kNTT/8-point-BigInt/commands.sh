#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=22,3 --fabric-offsets=8,1 \
--params=width:8 --colors=LAUNCH:8 --memcpy --channels=1 -o out-ntt
cs_python run.py --name out-ntt
