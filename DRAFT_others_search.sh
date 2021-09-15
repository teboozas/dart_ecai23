
# for seed in {0..100};do
#     python DRAFT_others.py --wandb --dataset=metabric --start_fold=0 --end_fold=1 --seed=$seed
# done

# for seed in {0..100};
# do
#     python DRAFT_others.py --wandb --dataset=metabric --start_fold=1 --end_fold=2 --seed=$seed
# done

# for seed in {0..100};
# do
#     python DRAFT_others.py --wandb --dataset=metabric --start_fold=2 --end_fold=3 --seed=$seed
# done

for seed in {0..100};
do
    python DRAFT_others.py --wandb --dataset=metabric --start_fold=3 --end_fold=4 --seed=$seed
done

# for seed in {0..100};
# do
#     python DRAFT_others.py --wandb --dataset=metabric --start_fold=4 --end_fold=5 --seed=$seed
# done 