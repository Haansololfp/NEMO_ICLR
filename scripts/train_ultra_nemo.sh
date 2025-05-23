python celltype_ibl/models/bimodal_embedding_main.py \
    --n_runs 100 \
    --log_every_n_steps 50 \
    --dataset Ultra \
    --test_data Ultra \
    -k 5 \
    -k 10 \
    -e 6000 \
    --adjust_to_ultra \
    --seed $seed
\