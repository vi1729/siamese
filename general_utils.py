import re

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from scipy import ndimage as ndi


def model_input_dim(x, scale=16):
    return x if x % scale == 0 else x + (scale - x % scale)


def resize_img(img, new_h, new_w):
    h, w = np.shape(img)[:2]
    if h == new_h and w == new_w:
        img = img.reshape(1, new_h, new_w,1)
    elif h > new_h or w > new_w:
        raise ValueError
    else:
        img2 = np.zeros((1, new_h, new_w, 1), dtype=np.float32)
        img2[0, :img.shape[0], :img.shape[1], 0] = img
        img = img2
    return img


def get_img_size(dataset_path, sequence):
    sample_img_path = glob.glob(os.path.join(dataset_path, "0{}\\*.tif".format(sequence)))[0]
    sample_img = cv2.imread(sample_img_path)
    return model_input_dim(np.shape(sample_img)[0]), model_input_dim(np.shape(sample_img)[1])


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


def make_list_of_imgs_only(imgs, img_extension):
    return [x for x in imgs if x.split('.')[-1] == img_extension]


def reshape_batch(img_batch, w, h):
    img_batch_new = []
    for im in img_batch:
        pad_w = int(w - im.shape[0])
        pad_h = int(h - im.shape[1])
        left_p = int(pad_w//2)
        right_p = pad_w - left_p
        top_p = int(pad_h//2)
        bottom_p = pad_h - top_p

        im_zeros = torch.zeros(w, h)
        im_zeros[max(0, left_p):min(w, w - right_p), max(0, top_p):min(h, h - bottom_p)] = im[max(0, -left_p):min(
            im.shape[0], im.shape[0] + right_p), max(0, -top_p):min(im.shape[1], im.shape[1] + bottom_p)]
        img_batch_new.append(im_zeros)
    return img_batch_new


def load_images(dataset_path, sequence, extension):
    extension_text = ".{}".format(extension)
    seg_dir = "\\0{}".format(sequence)
    imgs = dict()
    print("Loading images from {}".format(os.path.join(dataset_path + seg_dir, "*" + extension_text)))
    for img_path in glob.glob(os.path.join(dataset_path + seg_dir, "*" + extension_text)):
        name = img_path.split("/t")[-1].split(extension_text)[0]
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        imgs[name] = img
    return imgs


def get_frame(segmentation, generate_cell_labels=False):
    if generate_cell_labels:
        frame, _ = ndi.label(segmentation)
        return frame
    else:
        return segmentation


