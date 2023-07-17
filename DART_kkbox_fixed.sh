# export KAGGLE_USERNAME=junhyunlee
# export KAGGLE_KEY=419367c5a07c00793bc6f49e4af43bf8

for i in {0..33}; do 
python DART_kkbox.py --use_BN --seed=$i --dataset=kkbox_v1 --wandb;
done

# lr=(1e-1 1e-2 5e-3 1e-3 5e-4 1e-4)
# weight_decay=(0.4 0.2 0.1 0.05 0.02 0.01 0)
# an=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1.0)

# RANGE_lr=${#lr[@]}
# RANGE_weight_decay=${#weight_decay[@]}
# RANGE_an=${#an[@]}

# for (( i = 0 ; i < 10 ; i++ )) ; do
#     for loss in rank mae rmse; do
#         number_lr=$RANDOM
#         number_weight_decay=$RANDOM
#         number_an=$RANDOM
#         let "number_lr %= $RANGE_lr"
#         let "number_weight_decay %= $RANGE_weight_decay"
#         let "number_an %= $RANGE_an"
#         for num_layers in 6 4 2
#             do
#             for num_nodes in 256 128 512
#                 do
#                 for dropout in 0.1 0.0 0.5
#                     do
#                         # python search.py --dataset=kkbox --loss=$loss --optimizer=AdamWR --lr=${lr[$number_lr]} --weight_decay=${weight_decay[$number_weight_decay]} --an=${an[$number_an]} --num_layers=$num_layers --num_nodes=$num_nodes --dropout=$dropout --batch_size=4096 --use_BN 
#                     python search.py --dataset=kkbox_v2 --loss=$loss --optimizer=AdamWR --lr=0.0001 --weight_decay=0.0 --an=0.001 --num_layers=$num_layers --num_nodes=$num_nodes --dropout=$dropout --batch_size=1024 --use_BN 
#                 done
#             done
#         done
#     done
# done

