


# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0 > log_dmf_l2.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0001 >> log_dmf_l2.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0005 >> log_dmf_l2.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.001 >> log_dmf_l2.txt



# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 1 --l2 0.0 > log_dmf_ns.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 2 --l2 0.0 >> log_dmf_ns.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 3 --l2 0.0 >> log_dmf_ns.txt
# python -u run.py --model dmf --opt adam --lr 0.0005  --ns 4 --l2 0.0 >> log_dmf_ns.txt





python -u run.py --model dmf --opt adgrad --lr 0.0001  --ns 1 --l2 0.0 > log_dmf_opt_adg.txt
python -u run.py --model dmf --opt adgrad --lr 0.0005  --ns 1 --l2 0.0 >> log_dmf_opt_adg.txt
python -u run.py --model dmf --opt adgrad --lr 0.001   --ns 1 --l2 0.0 >> log_dmf_opt_adg.txt
python -u run.py --model dmf --opt adgrad --lr 0.005   --ns 1 --l2 0.0 >> log_dmf_opt_adg.txt

python -u run.py --model dmf --opt adadelta --lr 0.0001  --ns 1 --l2 0.0 > log_dmf_opt_adad.txt
python -u run.py --model dmf --opt adadelta --lr 0.0005  --ns 1 --l2 0.0 >> log_dmf_opt_adad.txt
python -u run.py --model dmf --opt adadelta --lr 0.001   --ns 1 --l2 0.0 >> log_dmf_opt_adad.txt
python -u run.py --model dmf --opt adadelta --lr 0.005   --ns 1 --l2 0.0 >> log_dmf_opt_adad.txt


