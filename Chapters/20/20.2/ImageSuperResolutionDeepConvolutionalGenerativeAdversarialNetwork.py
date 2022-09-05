'''

ResNet: Deep Residual Networks

'''

python3 srez_main.py --run train --batch_size 16 --checkpoint_dir ../Checkpoint --checkpoint_period 100 --dataset dataset --epsilon 1e-8 --gene_l1_factor 0.90 --learning_beta1 0.5 --learning_rate_start 0.00020 --learning_rate_life 5000 --sample_size 168 --summary_period 10 --random_seed 0 --test_vectors 16 --random_seed 0 --test_vectors 16 --train_dir Trains --train_time 20
