#!/usr/bin/env bash




python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 1 --test2 0 --neg_start 6 >> hope_deep_test.txt
python -u run.py --model dmf --opt adam --lr 0.0005 --ns 1 --test1 1 --test2 0 --neg_start 12 >> hope_deep_test.txt








