for i in {0..32}; do 
python DEEPHIT_kkbox.py --use_BN --seed=$i --dataset=kkbox_v1 --wandb;
done