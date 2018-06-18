

python -u run.py --model dmf --opt adam --lr 0.0001  --ns 1 --l2 0.0 > log_dmf_opt.txt
python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adam --lr 0.001   --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adam --lr 0.005   --ns 1 --l2 0.0 >> log_dmf_opt.txt


python -u run.py --model dmf --opt adgrad --lr 0.0001  --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adgrad --lr 0.0005  --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adgrad --lr 0.001   --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adgrad --lr 0.005   --ns 1 --l2 0.0 >> log_dmf_opt.txt

python -u run.py --model dmf --opt adadelta --lr 0.0001  --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adadelta --lr 0.0005  --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adadelta --lr 0.001   --ns 1 --l2 0.0 >> log_dmf_opt.txt
python -u run.py --model dmf --opt adadelta --lr 0.005   --ns 1 --l2 0.0 >> log_dmf_opt.txt


