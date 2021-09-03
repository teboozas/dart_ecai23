for layer in 4 6 8;
do
    for nodes in 128 256 512;
    do
        for dropout in 1.0 0.9 0.5;
            do
                python DRAFT_kkbox.py --dataset=kkbox_v1 --keep_prob=$dropout --num_nodes=$nodes --num_layers=$layer --wandb

            done
    done
done
