for intbit_w in 1 2 4 6; do
    for intbit_x in 1 2 4 6; do
        python train_quantization.py --model efficientnet --n_epoch 100 --print_freq 10 --lr 1e-4 --quant_w --bitwidth_w 8 --intbit_w ${intbit_w} --quant_x --bitwidth_x 8 --intbit_x ${intbit_x} 
    done
done