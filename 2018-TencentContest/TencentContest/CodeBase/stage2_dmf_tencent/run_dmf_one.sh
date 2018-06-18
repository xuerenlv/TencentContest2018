




python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0001 --uk 1 > sdmf_dmfdet_logtest.txt
python -u run.py --model dmf --opt adam --lr 0.001  --ns 1 --l2 0.0001 --uk 1 >> sdmf_dmfdet_logtest.txt
python -u run.py --model dmf --opt adam --lr 0.005  --ns 1 --l2 0.001 --uk 1 >> sdmf_dmfdet_logtest.txt


