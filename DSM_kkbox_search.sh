export CUDA_VISIBLE_DEVICES=0

for layer in 4 6 8;
do
    for nodes in 128 256 512;
    do
        for dropout in 0.0;
            do
                for dist in 'LogNormal' 'Weibull';
                do
                python DSM_kkbox.py --dataset=kkbox_v1 --dropout=$dropout --num_nodes=$nodes --num_layers=$layer --distribution=$dist --batch_size=1024 --wandb
                done
            done
    done
done
