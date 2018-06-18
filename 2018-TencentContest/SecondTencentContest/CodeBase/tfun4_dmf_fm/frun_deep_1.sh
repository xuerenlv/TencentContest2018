#!/usr/bin/env bash




python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 11 >> fhope_deep_1.txt
python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 10 >> fhope_deep_1.txt
python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 9 >> fhope_deep_1.txt
python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 0 --test2 1 --neg_start 8 >> fhope_deep_1.txt







