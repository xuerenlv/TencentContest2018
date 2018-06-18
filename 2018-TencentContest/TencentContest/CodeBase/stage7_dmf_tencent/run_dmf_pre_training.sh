





python -u run.py --model dmf --opt adam --lr 0.0001 --ns 1 --l2 0.0 --drk 1.0  --uk 1  --mtyp 4 \
--mp_w  /home/xuehj/TencentContest/CodeBase/stage6_dmf_tencent/checkpoints/dmf_tencent/mtyp2-2018-05-06-01-11-44 \
--mp_d  /home/xuehj/TencentContest/CodeBase/stage6_dmf_tencent/checkpoints/dmf_tencent/mtyp1-2018-05-06-00-30-21 \
>> hope_pre_training_1.txt

python -u run.py --model dmf --opt adam --lr 0.0001 --ns 1 --l2 0.0 --drk 1.0  --uk 1  --mtyp 4 \
--mp_w  /home/xuehj/TencentContest/CodeBase/stage6_dmf_tencent/checkpoints/dmf_tencent/mtyp2-2018-05-06-01-11-44 \
--mp_d  /home/xuehj/TencentContest/CodeBase/stage6_dmf_tencent/checkpoints/dmf_tencent/mtyp1-2018-05-06-00-30-21 \
>> hope_pre_training_1.txt







