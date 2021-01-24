import sys
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, io
import cv2
import matplotlib
import math
from sklearn.utils import class_weight
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import Sequence
import warnings
from keras import backend as K
from tensorflow.python.ops import control_flow_ops

from model_utils import unet_model, get_img_lbl_net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def main():
    # Hyper parameters
    batch_size = 3
    data_aug = True
    epochs = 10
    num_workers = 1
    aug_num = 15
    val_split = 0.1
    steps_per_epoch = ((np.ceil(aug_num / batch_size) if data_aug else 0) + 1) * num_workers
    print("Steps per epoch {}".format(steps_per_epoch))

    img_extension = 'tif'
    unet_path = None  #"model_weights/unet_model.h5"

    # dataset_paths_all = ["Fluo-N2DH-SIM+"]
    dataset_paths_all = ["DIC-C2DH-HeLa"]
    run_main(dataset_paths_all, unet_path, img_extension, val_split, batch_size, epochs, steps_per_epoch, data_aug,
             num_workers, aug_num)


def data_generator_infinite(imgs_paths, lbls_paths, new_h, new_w, data_aug, batch_size, aug_num):
    # epoch = 0
    while True:
        # epoch = epoch + 1
        shuffled_indices = np.random.permutation(len(imgs_paths))
        for i in shuffled_indices:
            img_path = imgs_paths[i]
            lbl_path = lbls_paths[i]

            img_net, lbl_net = get_img_lbl_net(img_path, new_h, new_w, lbl_path=lbl_path)

            # Data augmentation:
            if data_aug:
                aug_imgs, aug_lbls = get_augments(img_net, lbl_net, aug_num)
                for i_aug in range(0, len(aug_imgs), batch_size):
                    batch_imgs = aug_imgs[i_aug:i_aug + batch_size]
                    batch_lbls = aug_lbls[i_aug:i_aug + batch_size]
                    yield np.array(batch_imgs), np.array(batch_lbls)

            yield img_net, lbl_net


def get_new_img_edge_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def natural_keys(text):
    """
    Returns the key list for float numbers in file names for more proper sorting
    """

    def atof(text):
        """
        Used for sorting numbers, in float format
        """
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def get_augments(img, lbl, aug_num):
    imgs = []
    lbls = []
    noisy_aug = int(aug_num / 3)
    range_shift = int(aug_num / 3)
    range_chunk = aug_num - range_shift - noisy_aug
    for a in np.linspace(0.1, np.max(img), noisy_aug):
        imgs.extend(img + np.random.normal(0, a, np.prod(np.shape(img))).reshape(img.shape))
        lbls.extend(lbl)
    for a in range(range_shift):
        imgs.extend(img + 2 * (np.random.rand() - 0.5))
        lbls.extend(lbl)
    for a in range(range_chunk):
        max_val = np.random.rand() * (max(np.abs(np.min(img)), np.max(img)))
        min_or_max = np.random.rand() > 0.5
        imgs.extend(
            np.minimum(img, (np.max(img) - max_val)) if min_or_max else np.maximum(img, (max_val + np.min(img))))
        lbls.extend(lbl)
    return imgs, lbls


def run_main(dataset_paths_all, unet_path, img_extension, val_split, batch_size, epochs, steps_per_epoch, data_aug,
             num_workers, aug_num):
    for index_dataset_paths in range(len(dataset_paths_all)):
        dataset_paths = dataset_paths_all[index_dataset_paths:index_dataset_paths + 1]
        model_weights_output_filename = 'model_weights\\retrained_unet_{}_Jan2021.h5'.format(dataset_paths[0])

        print("Getting weights from {} and outputing results in {}".format(unet_path, model_weights_output_filename))
        print("{} datasets: {}".format(len(dataset_paths), dataset_paths))

        imgs_paths = []
        lbls_paths = []
        sample_imgs = []
        seqs = ["01", "02"]

        val_paths = []  # For harded images (located at the end of the second sequences
        for dataset in dataset_paths:
            for seq in seqs:
                label_paths = glob.glob(
                    '{}\*.{}'.format(os.path.join(dataset, "{}_GT".format(seq), 'TRA'), img_extension))
                images_paths = glob.glob('{}\*.{}'.format(os.path.join(dataset, seq), img_extension))
                print('*******************')
                print(images_paths)
                print(label_paths)
                lbls_available = sorted(label_paths, key=natural_keys)
                lbls_available_str = [x.split('\\')[3].replace("man_track", '') for x in lbls_available]

                imgs_to_add = sorted(images_paths, key=natural_keys)
                print(imgs_to_add)
                imgs_to_add_new = []
                for i, img_candidate in enumerate(imgs_to_add):
                    img_candidate_str = img_candidate.split('\\')[2][1:]
                    print(img_candidate_str)
                    if img_candidate_str in lbls_available_str:
                        imgs_to_add_new.append(img_candidate)
                        print(lbls_available_str)

                imgs_paths.extend(imgs_to_add_new)
                lbls_paths.extend(lbls_available)
                sample_imgs.append(cv2.imread(imgs_paths[-1], cv2.IMREAD_ANYDEPTH))
            val_paths.append(zip(imgs_paths[-5:], lbls_paths[-5:]))

        for i, img in enumerate(imgs_paths):
            lbl = lbls_paths[i]
            str_img = img.split('\\')
            lbl_str = lbl.split('\\')
            assert str_img[0] == lbl_str[0]
            assert str_img[1] in lbl_str[1]
            assert str_img[2][1:] == lbl_str[3].replace("man_track", '')

        max_w = -1
        max_h = -1
        for img in sample_imgs:
            h, w = np.shape(img)[:2]
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w
        del sample_imgs
        new_h = get_new_img_edge_value(max_h)
        new_w = get_new_img_edge_value(max_w)
        print('new_h   new_w')
        print(new_h,new_w)
        shuffled_indices = np.random.permutation(len(imgs_paths))
        train_indexes = shuffled_indices[int(len(shuffled_indices) * val_split):]
        val_indexes = shuffled_indices[:int(len(shuffled_indices) * val_split)]
        assert not bool(set(train_indexes) & set(val_indexes))

        train_imgs = [imgs_paths[i] for i in train_indexes]
        train_lbls = [lbls_paths[i] for i in train_indexes]
        val_imgs = [imgs_paths[i] for i in val_indexes]
        val_lbls = [lbls_paths[i] for i in val_indexes]
        assert not bool(set(train_imgs) & set(val_imgs))
        assert not bool(set(train_lbls) & set(val_lbls))
        print("Len imgs: {}. {} train. {} val".format(len(imgs_paths), len(train_imgs), len(val_imgs)))
        class_counts = {0: 0, 1: 0}
        for lbl_path in lbls_paths:
            lbl = ((cv2.imread(lbl_path, cv2.IMREAD_ANYDEPTH) > 0) * 1.0)
            class_counts[0] += np.sum(lbl == 0)
            class_counts[1] += np.sum(lbl == 1)
        a = 1 / (batch_size * (class_counts[0] / len(lbls_paths)))
        b = 1 / (batch_size * (class_counts[1] / len(lbls_paths)))
        class_weights = [a / (a + b), b / (a + b)]

        for val_i in val_paths:
            for couple in list(val_i):
                val_imgs.append(couple[0])
                val_lbls.append(couple[1])

        print("class_weights: {}".format(class_weights))
        print(len(train_imgs), len(train_lbls), aug_num)
        train_data_generator = data_generator_infinite(train_imgs, train_lbls, new_h, new_w, data_aug,
                                                       batch_size=batch_size, aug_num=aug_num)

        val_data_generator = data_generator_infinite(val_imgs, val_lbls, new_h, new_w, data_aug=True,
                                                     batch_size=batch_size, aug_num=aug_num)
        model = unet_model((new_h, new_w), unet_path)

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer='Adam')  # 'sparse_categorical_crossentropy'
        mcp_save = ModelCheckpoint(model_weights_output_filename, save_best_only=True,
                                   monitor='val_accuracy' if val_split > 0 else 'accuracy', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='acc', factor=0.0001, patience=3, verbose=1, epsilon=1e-4,
                                           mode='max')

        hist = model.fit_generator(train_data_generator, callbacks=[mcp_save, reduce_lr_loss], epochs=epochs,
                                   steps_per_epoch=len(train_imgs) * steps_per_epoch, verbose=1,
                                   class_weight=class_weights,
                                   validation_data=val_data_generator if val_split > 0 else None,
                                   validation_steps=len(val_imgs))


if __name__ == "__main__":
    main()
