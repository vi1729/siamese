from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model
import cv2
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
import os
from skimage.measure import regionprops
import copy
from tqdm import tqdm as tqdm
import shutil
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker

from general_utils import model_input_dim, resize_img


def unet_model(img_shape, path_to_weights=None, n_dims=1):

    in_layer = Input(shape=(img_shape[0], img_shape[1], n_dims))
    conv2d_1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(in_layer)
    conv2d_2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_1)
    drop1 = Dropout(0.5)(conv2d_2)
    maxpool_1 = MaxPooling2D((2, 2), padding='same')(drop1)
    conv2d_3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool_1)
    conv2d_4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_3)
    maxpool_2 = MaxPooling2D((2, 2), padding='same')(conv2d_4)
    conv2d_5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool_2)
    conv2d_6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_5)
    maxpool_3 = MaxPooling2D((2, 2), padding='same')(conv2d_6)
    conv2d_7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool_3)
    conv2d_8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_7)
    drop2 = Dropout(0.5)(conv2d_8)
    maxpool_4 = MaxPooling2D((2, 2), padding='same')(drop2)

    conv2d_9 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(maxpool_4)
    conv2d_10 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_9)
    drop3 = Dropout(0.5)(conv2d_10)

    upsample_1 = UpSampling2D((2, 2), interpolation='bilinear')(drop3)
    concat_1 = Concatenate(axis=3)([upsample_1, drop2])

    conv2d_11 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concat_1)
    conv2d_12 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_11)
    upsample_2 = UpSampling2D((2, 2), interpolation='bilinear')(conv2d_12)
    concat_2 = Concatenate(axis=3)([upsample_2, conv2d_6])

    conv2d_13 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concat_2)
    conv2d_14 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_13)
    upsample_3 = UpSampling2D((2, 2), interpolation='bilinear')(conv2d_14)
    concat_3 = Concatenate(axis=3)([upsample_3, conv2d_4])

    conv2d_15 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concat_3)
    conv2d_16 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_15)
    upsample_4 = UpSampling2D((2, 2), interpolation='bilinear')(conv2d_16)
    concat_4 = Concatenate(axis=3)([upsample_4, drop1])

    conv2d_17 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concat_4)
    conv2d_18 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2d_17)
    conv2d_19 = Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2d_18)
    model = Model(in_layer, conv2d_19)
    print('path_to_weights')
    print(path_to_weights)
    if path_to_weights is not None:
        model.load_weights(path_to_weights)

    return model



def get_img_lbl_net(img_path, new_h, new_w, lbl_path=None):

    if type(img_path) == str:
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    else:
        img = img_path
    # take ln of image
    img_log = np.log(np.float64(img), dtype=np.float64)

    # do dft saving as complex output
    dft = np.fft.fft2(img_log, axes=(0, 1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    # create black circle on white background for high pass filter
    radius = 5
    mask = np.zeros_like(img, dtype=np.float64)
    cy = mask.shape[0]
    cx = mask.shape[1]
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    mask = 1 - mask

    # antialias mask via blurring
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    # mask = cv2.GaussianBlur(mask, (47,47), 0)

    # apply mask to dft_shift
    dft_shift_filtered = np.multiply(dft_shift, mask)

    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift_filtered)

    # do idft saving as complex
    img_back = np.fft.ifft2(back_ishift, axes=(0, 1))

    # combine complex real and imaginary components to form (the magnitude for) the original image again
    img_back = np.abs(img_back)

    # apply exp to reverse the earlier log
    # img = np.exp(img_back, dtype=np.float64)

    # def _histeq(im, nbr_bins=256):
    #     imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    #     cdf = imhist.cumsum()  # cumulative distribution function
    #     cdf = 255 * cdf / cdf[-1]  # normalize
    #
    #     # use linear interpolation of cdf to fnp.log(img+1)ind new pixel values
    #     im2 = np.interp(im.flatten(), bins[:-1], cdf)
    #
    #     return im2.reshape(im.shape), cdf
    #
    # if type(img_path) == str:
    #     img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    # else:
    #     img = img_path
    #
    # img_log = np.log(img + (1.1-np.min(img)))
    # img, _ = _histeq(img_log)
    img = img_back / np.max(img_back)
    img = (img - np.mean(img)) / np.std(img)
    img_net = np.zeros((1, new_h, new_w, 1), dtype=np.float32) + np.min(img)
    img_net[0, :img.shape[0], :img.shape[1], 0] = img

    lbl_net = None
    if lbl_path is not None:
        lbl = (cv2.imread(lbl_path, cv2.IMREAD_ANYDEPTH) > 0) * 1.0
        lbl_net = np.zeros((1, new_h, new_w, 1), dtype=np.float32)
        lbl_net[0, :img.shape[0], :img.shape[1], 0] = lbl

    return img_net, lbl_net


def get_segmentation(input_img, model):
    img_shape = np.shape(input_img)
    new_h, new_w = model_input_dim(img_shape[0]), model_input_dim(img_shape[1])

    img_net, _ = get_img_lbl_net(input_img, new_h, new_w)

    prediction = model.predict(img_net, batch_size=1)
    pred = prediction[0, :img_shape[0], :img_shape[1], 0] >= 0.5
    return pred


def get_labels(prediction):
    def _get_reduced_maxima(pred):
        markers = []
        distance = ndi.distance_transform_edt(pred)
        distance_ordinal = distance - np.min(distance)
        distance_ordinal = distance_ordinal / np.max(distance_ordinal)
        markers.append(ndi.label(distance_ordinal)[0])
        markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.1 * 1.0))[0])
        # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.5 * 1.0))[0])
        # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.75 * 1.0))[0])
        final_markers = ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.5 * 1.0))[0]

        # # Calculate mean cell size
        cell_sums = []
        for c_id in np.unique(final_markers)[1:]:
            cell_sums.append(np.sum(c_id == final_markers * 1.0))
        mean_sum = np.mean(cell_sums)

        # Reduce centroid size
        for marker in markers[::-1]:
            transition_markers = marker * (final_markers > 0)
            for c_id in np.unique(marker)[1:]:
                cell_seg = (marker == c_id) * 1.0
                if np.sum(cell_seg * (transition_markers > 0 * 1.0)) <= mean_sum * 0.3:
                    final_markers = ndi.label((final_markers > 0)*1.0 + (marker == c_id)*1.0)[0]

        return final_markers, mean_sum

    pred, ind = get_expanded_boundary(prediction)
    markers, mean_sum = _get_reduced_maxima(pred)

    markers = get_normal_boundary(markers, ind)
    markers = markers * (prediction > 0 * 1.0) - 1 * (prediction == 0 * 1.0)
    wa = random_walker(prediction, markers)
    wa = wa * (wa != -1)

    # Remove small cell artifacts
    wa_2 = np.zeros(np.shape(wa))
    for c_id in np.unique(wa)[1:]:
        if np.sum((c_id == wa) * 1.0) >= mean_sum*0.1:
            wa_2 += (c_id == wa) * c_id
    wa = wa_2
    return wa


# def get_labels(prediction):
#     def _get_mean(final_markers):
#         cell_sums = []
#         for c_id in np.unique(final_markers)[1:]:
#             cell_sums.append(np.sum(c_id == final_markers * 1.0))
#         mean_sum = np.mean(cell_sums)
#         return mean_sum
#
#     def _get_reduced_maxima(pred):
#         markers = []
#         distance = ndi.distance_transform_edt(pred)
#
#         distance_ordinal = distance - np.min(distance)
#         distance_ordinal = distance_ordinal / np.max(distance_ordinal)
#         markers.append(ndi.label(distance_ordinal)[0])
#         markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.1 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.2 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.3 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.4 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.5 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.6 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.7 * 1.0))[0])
#         # markers.append(ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.8 * 1.0))[0])
#         final_markers = ndi.label(ndi.distance_transform_edt(distance_ordinal >= 0.5 * 1.0))[0]
#
#         # plt.subplot(1,3,1)
#         # plt.imshow(distance_ordinal)
#         # plt.subplot(1, 3, 2)
#         # plt.imshow(ndi.distance_transform_edt(distance_ordinal >= 0.1 * 1.0))
#         # plt.subplot(1, 3, 3)
#         # plt.imshow(ndi.distance_transform_edt(distance_ordinal >= 0.9 * 1.0))
#         # plt.show()
#         # assert False
#
#         # Calculate mean cell size
#
#         # Reduce centroid size
#         for marker in markers[::-1]:
#             mean_sum = _get_mean(marker)
#             # plt.subplot(1, 3, 1)
#             # plt.imshow(pred)
#             # plt.subplot(1,3,3)
#             # plt.imshow(final_markers)
#             # plt.title(len(np.unique(final_markers))-1)
#             # plt.subplot(1, 3, 2)
#             # plt.imshow(marker)
#             # plt.show()
#
#             transition_markers = marker * (final_markers > 0)
#             for c_id in np.unique(marker):
#                 if c_id == 0:
#                     continue
#                 cell_seg = (marker == c_id) * 1.0
#                 if np.sum(cell_seg) >= mean_sum * 0.1 and len(np.unique(cell_seg*final_markers)) <= 2:
#                 # if np.sum(cell_seg * (transition_markers > 0 * 1.0)) == 0 and np.sum(cell_seg) >= mean_sum * 0.1:
#                 # if np.sum(cell_seg * (transition_markers > 0 * 1.0)) <= mean_sum * 0.3:
#                     final_markers = ndi.label((final_markers > 0)*1.0 + (marker == c_id)*1.0)[0]
#
#         # plt.imshow(final_markers)
#         # plt.title(len(np.unique(final_markers)) - 1)
#         # plt.show()
#         # assert False
#         return final_markers
#
#     pred, ind = get_expanded_boundary(prediction)
#     markers = _get_reduced_maxima(pred)
#
#     markers = get_normal_boundary(markers, ind)
#     markers = markers * (prediction > 0 * 1.0) - 1 * (prediction == 0 * 1.0)
#     wa = random_walker(prediction, markers)
#     wa = wa * (wa != -1)
#
#     # Remove small cell artifacts
#     wa_2 = np.zeros(np.shape(wa))
#     mean_sum = _get_mean(wa)
#     for c_id in np.unique(wa):
#         if c_id == 0:
#             continue
#         if np.sum((c_id == wa) * 1.0) >= mean_sum*0.1:
#             wa_2 += (c_id == wa) * c_id
#     wa = wa_2
#     return wa


def get_expanded_boundary(cell_seg, edge_leeway=20):
    distance = np.zeros((np.shape(cell_seg)[0] + edge_leeway, np.shape(cell_seg)[1] + edge_leeway))
    ind = [int(edge_leeway / 2), -int(edge_leeway / 2), int(edge_leeway / 2), -int(edge_leeway / 2)]
    distance[ind[0]:ind[1], ind[2]:ind[3]] = cell_seg
    return distance, ind


def get_normal_boundary(cell_seg, ind):
    return cell_seg[ind[0]:ind[1], ind[2]:ind[3]]


def argmax_2d(matrix):
    maxN = np.argmax(matrix)
    (xD,yD) = matrix.shape
    if maxN >= xD:
        x = maxN//xD
        y = maxN % xD
    else:
        y = maxN
        x = 0
    return (x,y)


def resegment_frame(frame, cell_id, matched_cells):

    def _get_centroid(matched_cell_seg):
        regions = regionprops(matched_cell_seg.astype(int))
        assert len(regions) == 1 and len(regions[0].bbox) == 4
        return regions[0].centroid

    cell_seg = (frame == cell_id)*1.0
    new_frame = copy.deepcopy(frame)
    new_frame = ((cell_seg == 0) * 1) * new_frame

    distance = copy.deepcopy(cell_seg)
    markers = np.zeros((np.shape(cell_seg)))
    new_cell_seg = copy.deepcopy(cell_seg)

    for c_id, matched_cell_dict in enumerate(matched_cells):
        distance_cell = matched_cell_dict['orig_cell_seg']
        distance_cell = ndi.distance_transform_edt(distance_cell)
        distance += distance_cell
        y, x = _get_centroid(matched_cell_dict['orig_cell_seg'])

        if not matched_cell_dict['orig_cell_seg'][int(y), int(x)] >= 1:
            print("No centroid found for cell with id {}".format(matched_cell_dict['id']))
            markers += (matched_cell_dict['orig_cell_seg'] > 0) * (max(np.unique(frame)) + c_id + 1)
            continue

        if not matched_cell_dict['orig_cell_seg'][int(y), int(x)] >= 1:
            print(matched_cells[0]['id'])
            print(matched_cells[1]['id'])
            assert matched_cell_dict['orig_cell_seg'][int(y), int(x)] >= 1
        markers[int(y), int(x)] += max(np.unique(frame)) + c_id + 1

    new_cell_seg = watershed(-distance, markers, mask=new_cell_seg)
    new_frame = new_frame + new_cell_seg

    return new_frame


def prepare_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)


def do_siamese_tracking(frames, imgs, sorted_img_indx, track_model, args):

    print("Running Siamese tracker")
    imgs_directory = os.path.join(args.dataset_path, "0{}_RES".format(args.sequence))
    dir(imgs_directory)

    number_of_mitosis = 0
    number_of_collisions = 0
    width, height = frames[sorted_img_indx[0]].shape
    previous_frame = np.zeros((width, height))
    cells = dict()
    video_frames = []
    for current_frame_id, img_indx in tqdm(enumerate(sorted_img_indx)):
        frame = frames[img_indx]

        # Collision detection
        if args.do_collision_tracking:
            img_2 = None if current_frame_id == 0 else np.stack(
                (imgs[sorted_img_indx[current_frame_id - 1]].astype(np.int16),) * 3, axis=-1)
            if img_2 is not None:
                frame_2 = previous_frame
                img_1 = np.stack((imgs[img_indx].astype(np.int16),) * 3, axis=-1)
                frame_1 = frames[img_indx]
                c_ids = np.unique(frame_1)
                for cell_id in c_ids:
                    if cell_id == 0:
                        continue
                    # Backwards pass
                    len_matched_prev_frame, matched_cells = get_len_matched_next_frame(frame_1, frame_2, cell_id,
                                                                                       movement_threshold=args.movement_threshold,
                                                                                       iou_2_cell_thresh=args.iou_2_cell_thresh,
                                                                                       img_1=img_1, img_2=img_2,
                                                                                       track_model=track_model,
                                                                                       return_matched_cells=True)

                    if len_matched_prev_frame == 2:
                        # Re-segment
                        number_of_collisions += 1
                        matched_cells[0]['orig_cell_seg'] = (frame_2 == matched_cells[0]['id']) * 1.0
                        matched_cells[1]['orig_cell_seg'] = (frame_2 == matched_cells[1]['id']) * 1.0
                        frame_1 = resegment_frame(frame_1, cell_id, matched_cells)
                frame = frame_1

        img_2 = np.stack((imgs[img_indx].astype(np.int16),) * 3, axis=-1)
        img_1 = None if current_frame_id == 0 else np.stack(
            (imgs[sorted_img_indx[current_frame_id - 1]].astype(np.int16),) * 3, axis=-1)
        img_3 = None if (current_frame_id + 1) >= len(sorted_img_indx) else np.stack(
            (imgs[sorted_img_indx[current_frame_id + 1]].astype(np.int16),) * 3, axis=-1)
        next_frame = np.zeros((width, height), np.uint16)
        encountered_cells = []

        # Mitosis detection
        c_ids = np.unique(frame)
        for cell_id in c_ids:
            if cell_id == 0:
                continue
            cell_segmentation, matched_cell_id, iou = get_matched_cell(frame, previous_frame, cell_id,
                                                                       movement_threshold=args.movement_threshold,
                                                                       img_1=img_2, img_2=img_1,
                                                                       track_model=track_model)

            if iou > args.img_bb_threshold:
                if matched_cell_id not in encountered_cells:
                    # Exact match: 1 <-> 1 (assuming)
                    cells[matched_cell_id].last_frame = current_frame_id
                    next_frame_cell_id = matched_cell_id
                else:
                    # Forwards pass: Check next frame
                    len_matched_next_frame = None
                    if not (current_frame_id+1 == len(sorted_img_indx)) and args.do_one_step_look_ahead:
                        len_matched_next_frame, matched_cells = get_len_matched_next_frame(previous_frame, frames[
                            sorted_img_indx[current_frame_id + 1]], matched_cell_id,
                                                                                           movement_threshold=args.movement_threshold,
                                                                                           iou_2_cell_thresh=args.iou_2_cell_thresh,
                                                                                           img_1=img_1, img_2=img_3,
                                                                                           track_model=track_model,
                                                                                           return_matched_cells=True)

                    if (len_matched_next_frame is not None) and (
                            len_matched_next_frame == 1 or len_matched_next_frame == 0 or (
                            len_matched_next_frame == 2 and (
                            np.sum(matched_cells[0]['seg']) < 50 or np.sum(matched_cells[1]['seg']) < 50))):
                        # Re-segment current frame back to 1 cell
                        next_frame_cell_id = matched_cell_id
                    else:
                        # Mitosis: 1 -> 2
                        number_of_mitosis += 1
                        if cells[matched_cell_id].last_frame == current_frame_id:
                            cells[matched_cell_id].last_frame = current_frame_id - 1
                            cell_2_segmentation = (next_frame == matched_cell_id) * 1
                            next_frame = next_frame - cell_2_segmentation * matched_cell_id + cell_2_segmentation * (
                                        len(cells.keys()) + 1)
                            cells[len(cells.keys()) + 1] = Cell(len(cells.keys()) + 1, current_frame_id,
                                                                parent=int(matched_cell_id))

                        # create new record with parent cell max_candidate
                        cells[len(cells.keys()) + 1] = Cell(len(cells.keys()) + 1, current_frame_id,
                                                            parent=int(matched_cell_id))
                        next_frame_cell_id = len(cells.keys())

                encountered_cells.append(matched_cell_id)
            else:
                # No matched cell from previous frame -> cell birth
                len_matched_next_frame = None
                if not (current_frame_id + 1 == len(sorted_img_indx)) and args.do_one_step_look_ahead:
                    len_matched_next_frame = get_len_matched_next_frame(previous_frame, frames[
                        sorted_img_indx[current_frame_id + 1]], matched_cell_id,
                                                                           movement_threshold=args.movement_threshold,
                                                                           iou_2_cell_thresh=args.iou_2_cell_thresh,
                                                                           img_1=img_1, img_2=img_3,
                                                                           track_model=track_model)
                if (len_matched_next_frame is not None) and (len_matched_next_frame == 0):
                    # Was a false prediction
                    print("Ignoring cell")
                    continue
                cells[len(cells.keys()) + 1] = Cell(len(cells.keys()) + 1, current_frame_id)
                next_frame_cell_id = len(cells.keys())
            next_frame = next_frame + cell_segmentation * next_frame_cell_id

        img_path_name = os.path.join(imgs_directory, "mask{}.{}".format(img_indx, args.img_extension))
        cv2.imwrite(img_path_name, next_frame.astype(np.uint16))
        previous_frame = next_frame
        video_frames.append(previous_frame)

    print("number_of_mitosis: {}".format(number_of_mitosis))
    print("number_of_collisions: {}".format(number_of_collisions))

    if args.do_store_video:
        store_video(args.dataset_path, args.sequence, video_frames)

    # Store tracking
    with open(os.path.join(imgs_directory, 'res_track.txt'), "w") as tracks_file:
        print(imgs_directory)
        print("Saving tracking file")
        for cell in cells.values():
            tracks_file.write('{}\n'.format(cell.get_tracking_info()))


def get_len_matched_next_frame(next_frame, previous_frame, cell_id, movement_threshold, iou_2_cell_thresh,
                                   img_1=None, img_2=None, track_model=None, cell_segmentation=None,
                                   return_matched_cells=False):
    def _get_cell_dict(cell_segmentation, matched_cell_id, max_iou):
        matched_cell_dict = dict()
        matched_cell_dict['seg'] = cell_segmentation
        matched_cell_dict['id'] = matched_cell_id
        matched_cell_dict['iou'] = max_iou
        return matched_cell_dict

    cell_segmentation_1, matched_cell_id_1, max_iou_1 = get_matched_cell(next_frame, previous_frame, cell_id,
                                                                         movement_threshold=movement_threshold,
                                                                         img_1=img_1, img_2=img_2,
                                                                         track_model=track_model, cell_segmentation=cell_segmentation)
    cell_segmentation_2, matched_cell_id_2, max_iou_2 = get_matched_cell(next_frame, previous_frame, cell_id,
                                                                         movement_threshold=movement_threshold,
                                                                         ignore_id=matched_cell_id_1, img_1=img_1,
                                                                         img_2=img_2, track_model=track_model, cell_segmentation=cell_segmentation)
    matched_cells = None
    if return_matched_cells:
        matched_cells = [_get_cell_dict(cell_segmentation_1, matched_cell_id_1, max_iou_1),
                         _get_cell_dict(cell_segmentation_2, matched_cell_id_2, max_iou_2)]

    if max_iou_1 > iou_2_cell_thresh and max_iou_2 > iou_2_cell_thresh:
        return 2 if matched_cells is None else 2, matched_cells
    elif max_iou_1 == 0 and max_iou_2 == 0:
        return 0 if matched_cells is None else 0, matched_cells
    else:
        return 1 if matched_cells is None else 1, matched_cells


def get_matched_cell(frame, previous_frame, cell_id, movement_threshold, ignore_id=None, img_1=None, img_2=None,
                         track_model=None, cell_segmentation=None):
    if img_1 is not None and img_2 is not None and track_model is not None:
        new_frame_prediction, cell_segmentation = get_new_frame_prediction(img_1, img_2, track_model, frame, cell_id,
                                                                           movement_threshold=movement_threshold,
                                                                           cell_segmentation=cell_segmentation)
        union_region = np.maximum(new_frame_prediction, cell_segmentation) * previous_frame
    else:
        cell_segmentation = (frame == cell_id) * 1.0
        union_region = cell_segmentation * previous_frame

    max_iou, matched_cell_id = 0, 0

    for new_cell_id in np.unique(union_region)[1:]:
        if new_cell_id == 0 or ignore_id == new_cell_id:
            continue
        iou = np.sum(union_region == new_cell_id * 1.0) / np.sum(cell_segmentation)
        if iou > max_iou:
            max_iou = iou
            matched_cell_id = new_cell_id

    return cell_segmentation, matched_cell_id, max_iou


def get_new_frame_prediction(img_1, img_2, track_model, frame, cell_id, movement_threshold, cell_segmentation=None):
    def _get_new_dxdy(dx, max_len):
        if dx < 0:
            dx1 = 0
            dx2 = max_len + dx
            dx3 = -dx
            dx4 = max_len
        else:
            dx1 = dx
            dx2 = max_len
            dx3 = 0
            dx4 = max_len - dx
        return dx1, dx2, dx3, dx4

    if cell_segmentation is None:
        cell_segmentation = (frame == cell_id) * 1.0
    regions = regionprops(cell_segmentation.astype(int))
    try:
        min_row, min_col, max_row, max_col = regions[0].bbox
    except IndexError:
        plt.subplot(1,2,1)
        plt.imshow(frame)
        plt.title("Frame")
        plt.subplot(1, 2, 2)
        plt.imshow(cell_segmentation)
        plt.title("Cell segmentation")
        plt.show()
        raise IndexError
    track_model.init(img_1, [min_col, min_row, int(max_col - min_col), int(max_row - min_row)])
    [y, x, _, _] = track_model.update(img_2)

    dx = int(x - min_row)
    dy = int(y - min_col)
    if np.abs(dx) > movement_threshold:
        dx = 0
    if np.abs(dy) > movement_threshold:
        dy = 0
    dx_slices = _get_new_dxdy(dx, np.shape(cell_segmentation)[0])
    dy_slices = _get_new_dxdy(dy, np.shape(cell_segmentation)[1])

    predicted_cell_location = np.zeros(np.shape(cell_segmentation))
    predicted_cell_location[dx_slices[0]:dx_slices[1], dy_slices[0]:dy_slices[1]] = cell_segmentation[
                                                                                     dx_slices[2]:dx_slices[3],
                                                                                     dy_slices[2]:dy_slices[3]]
    return predicted_cell_location, cell_segmentation


def store_video(dataset_path, sequence, video_frames, fps=3):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("{}/result_0{}.avi".format(dataset_path, sequence), fourcc, float(fps),
                          (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame.astype(np.uint8))
    out.release()


class Cell:
    def __init__(self, cell_id, current_frame, parent=0):
        self.cell_id = cell_id
        self.first_frame = current_frame
        self.last_frame = current_frame
        self.parent = parent

    def __repr__(self):
        return self.get_id()

    def get_id(self):
        return self.cell_id

    def set_id(self, cell_id):
        self.cell_id = cell_id

    def get_tracking_info(self):
        return "{} {} {} {}".format(self.cell_id, self.first_frame, self.last_frame, self.parent)
