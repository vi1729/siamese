import argparse
from general_utils import get_img_size, natural_keys, make_list_of_imgs_only, get_frame, load_images
from siamfc import TrackerSiamFC
from model_utils import unet_model, get_segmentation, prepare_dir, do_siamese_tracking, get_labels
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import glob
import warnings
from initial_seg_lux import predict_dataset, predict_dataset_2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def main(args):
    print("Initialising Tracker and Segmentation network")
    img_shape = get_img_size(args.dataset_path, args.sequence)
    track_model = TrackerSiamFC(net_path=args.siamese_path, use_cuda=args.use_cuda)
    seg_model = unet_model(img_shape, args.unet_path)

    imgs = load_images(args.dataset_path, args.sequence, args.img_extension)
    sorted_img_indx = sorted(imgs.keys(), key=natural_keys)
    if args.debug or args.debug_v2:
        if args.debug:
            sorted_img_indx = sorted_img_indx[int(len(sorted_img_indx)/2):int(len(sorted_img_indx)/2)+3]
        else:
            sorted_img_indx = sorted_img_indx[:2]
        print("Debugging conditions set")
    frames = dict()

    if args.save_model_preds_to_file is not None:
        print("Creating '{}' to save model predictions".format(args.save_model_preds_to_file))
        args.save_model_preds_to_file = os.path.join(args.dataset_path, args.save_model_preds_to_file)
        prepare_dir(args.save_model_preds_to_file)

    if args.load_model_preds_from_file is not None:
        args.load_model_preds_from_file = os.path.join(args.dataset_path, args.load_model_preds_from_file)

    print("Getting segmentation")
    if args.get_initial_seg_from_lux:
        dataset_name = args.dataset_path.split('/')[-1] if not args.dataset_path[-1] == '/' else \
                       args.dataset_path.split('/')[
                           -2]
        print("Getting initial segmentation for {}".format(dataset_name))
        predict_dataset(name=dataset_name, sequence="0{}".format(args.sequence),
                        model_path='model_weights/unet_{}.h5'.format(dataset_name),
                        output_path="mulux_0{}".format(args.sequence))
        predict_dataset_2(path="mulux_0{}".format(args.sequence),
                          output_path="mulux_0{}".format(args.sequence))

        for img_indx_id, img_name in enumerate(sorted(glob.glob("mulux_0{}/mask*.{}".format(args.sequence, args.img_extension)), key=natural_keys)):
            frames[sorted_img_indx[img_indx_id]] = get_frame(cv2.imread(img_name, cv2.IMREAD_ANYDEPTH))
        # shutil.rmtree("mulux_0{}".format(args.sequence), ignore_errors=True)
    elif args.load_segs_from_file is None:
        for img_id, img_indx in enumerate(sorted_img_indx):
            if args.load_model_preds_from_file is not None:
                img_path = os.path.join(args.load_model_preds_from_file, "{}{}.tif".format(args.preds_extension, img_indx))
                pred = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                if pred is None or (pred is not None and np.sum(pred) == 0):
                    print(img_path)
                    assert pred is not None
            else:
                pred = get_segmentation(imgs[img_indx], seg_model)
            if args.load_preds_as_are:
                frames[img_indx] = pred
            else:
                frames[img_indx] = get_frame(get_labels(pred))
            if args.save_model_preds_to_file is not None:
                img_path_name = os.path.join(args.save_model_preds_to_file, "{}.tif".format(img_indx))
                cv2.imwrite(img_path_name, pred.astype(np.uint8))
    else:
        load_segs_dir = os.path.join(args.dataset_path, args.load_segs_from_file)
        imgs_list = sorted(make_list_of_imgs_only(os.listdir(load_segs_dir), args.img_extension),
                           key=natural_keys)
        print("Reading segs from: {}".format(load_segs_dir))
        for img_indx_id, img_name in enumerate(imgs_list):
            img_path = os.path.join(load_segs_dir, img_name)
            frames[sorted_img_indx[img_indx_id]] = get_frame(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH))

    if args.save_segs_to_file is not None:
        save_segs_dir = os.path.join(args.dataset_path, args.save_segs_to_file)
        prepare_dir(save_segs_dir)
        print("Saving segs to: {}".format(save_segs_dir))
        for img_id, img_indx in enumerate(sorted_img_indx):
            img_path_name = os.path.join(save_segs_dir, "{}.{}".format(img_indx, args.img_extension))
            cv2.imwrite(img_path_name, frames[img_indx].astype(np.uint8))
    exit()
    do_siamese_tracking(frames, imgs, sorted_img_indx, track_model, args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cell detection')
    parser.add_argument('--dataset_path', required=True, type=str,
                        help='Path to the dataset, for example: Fluo-N2DH-SIM+')
    parser.add_argument('--sequence', required=True, type=int,
                        help='sequence 1 or 2')
    parser.add_argument('-cpu', dest="use_cuda", action='store_false')
    parser.add_argument('--load_segs_from_file', type=str,
                        help='Name of the folder to load the segmented imgs')
    parser.add_argument('--save_segs_to_file', type=str,
                        help='Name of the folder to save the segmented imgs at')
    parser.add_argument('--load_model_preds_from_file', type=str,
                        help='Name of the folder to load the model prediction imgs')
    parser.add_argument('-load_preds_as_are', dest='load_preds_as_are', action='store_true')
    parser.add_argument('--preds_extension', type=str, default='')
    parser.add_argument('--save_model_preds_to_file', type=str,
                        help='Name of the folder to save the model prediction imgs at')
    parser.add_argument('--img_extension', type=str, default='tif')
    parser.add_argument('--img_bb_threshold', type=float, default=0.2)
    parser.add_argument('--movement_threshold', type=float, default=20)
    parser.add_argument('--iou_2_cell_thresh', type=float, default=0.1)
    parser.add_argument('--unet_path', type=str,
                        help='path to the unet weights')
    parser.add_argument('--siamese_path', type=str,
                        help='path to the siamese tracker weights')
    parser.add_argument('-no_one_step_look_ahead', dest='do_one_step_look_ahead', action='store_false')
    parser.add_argument('-no_collision_tracking', dest='do_collision_tracking', action='store_false')
    parser.add_argument('-do_store_video', action='store_true')
    parser.add_argument('-d', dest='debug', action='store_true')
    parser.add_argument('-d2', dest='debug_v2', action='store_true')

    parser.add_argument('-get_initial_seg_from_lux', action='store_true')

    args = parser.parse_args()

    main(args)
