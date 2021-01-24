#!/bin/bash

python -u siam_track.py --dataset_path="DIC-C2DH-HeLa" --sequence=1 --siamese_path="model_weights/siam_model.pth" -get_initial_seg_from_lux -cpu -do_store_video



python -u siam_track.py -d -d2 --dataset_path="DIC-C2DH-HeLa" --sequence=2 --siamese_path="model_weights/siam_model.pth" -get_initial_seg_from_lux --save_model_preds_to_file="02_SEG" --save_segs_to_file="02_SEG" -cpu