#!/usr/bin/env bash




python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 18 >> fhope_deep_1.txt
python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 15 >> fhope_deep_1.txt







