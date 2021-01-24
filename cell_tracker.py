from __future__ import absolute_import, division

from skimage.measure import regionprops
from skimage import measure
from skimage.segmentation import random_walker

import random
from tqdm import tqdm
from copy import deepcopy

bla_count = 0

from general_utils import natural_keys, make_list_of_imgs_only
from siamfc import TrackerSiamFC

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
from skimage.morphology import watershed
from scipy import ndimage as ndi

def get_seg(prediction, scale, par1, par2):
    naive_seg = cv2.inRange((prediction).astype(int), par1, scale)
    distance = ndi.distance_transform_edt(naive_seg)
    new_m = cv2.inRange((prediction).astype(int), par2, scale)
    _, markers = cv2.connectedComponents(new_m)
    return (watershed(-distance, markers, mask=naive_seg) > 0.0)*1.0


class Cell:
    def __init__(self, image, region):
        self.min_row, self.min_col, self.max_row, self.max_col = region.bbox
        self.tl = (self.min_col, self.min_row)
        self.br = (self.max_col, self.max_row)
        self.width = int(self.max_col - self.min_col)
        self.height = int(self.max_row - self.min_row)
        self.parent_image = image

        self.image = image[self.min_row:self.max_row, self.min_col:self.max_col]
        self.segmentation = region.image
        self.color = region.label
        self.centroid_y, self.centroid_x = region.centroid
        self.region = region

    def __repr__(self):
        return "(x: {}, y: {})".format(self.centroid_x, self.centroid_y)

    def plot(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.parent_image)
        plt.title("Input image")

        plt.subplot(1, 3, 2)
        plt.imshow(self.image)
        plt.title("Cell")

        plt.subplot(1, 3, 3)
        plt.imshow(self.segmentation)
        plt.title("Segmentation")
        plt.show()


class CellTrack:
    def __init__(self, cell, frame_count, _id, parent=0):
        self.alive = True
        self.cell_id = _id

        self.display_color = tuple(random.sample(range(0, 255), 3))
        self.parent = parent
        self.first_frame = frame_count
        self.last_frame = frame_count
        self.length = 1
        self.track = [cell]
        self.current_cell = cell

    def remove_last_cell(self):
        last_cell = self.track[-1]
        self.track = self.track[:-1]
        self.current_cell = self.track[-1]
        self.length -= 1
        self.last_frame -= 1
        return last_cell

    def add_cell(self, new_cell):
        self.track.append(new_cell)
        self.current_cell = new_cell
        self.length += 1
        self.last_frame += 1

    def __str__(self):
        return "{} {} {} {}".format(self.cell_id, self.first_frame, self.last_frame, self.parent)

    def __repr__(self):
        return "{} {} {} {}".format(self.cell_id, self.first_frame, self.last_frame, self.parent)


class CellImage:
    def __init__(self, image, image_name, dataset_path, sequence, seg_gt=None):
        self.sequence = sequence
        self.image_name = image_name
        self.frame_count = int(image_name)
        self.dataset_path = dataset_path
        self.image = image
        self.seg_gt = seg_gt
        self.binary_seg = None
        self.seg_output = None
        self.cell_count = None
        self.cell_locations = []

    def binary_seg_to_output(self):
        self.seg_output, self.cell_count = ndi.label(self.binary_seg)

    def get_cell_locations(self):
        return [Cell(self.image, region) for region in regionprops(self.seg_output)]

    # stores the segmentation mask
    def store(self):
        store_path = os.path.join(self.dataset_path, "0{}_RES/".format(self.sequence))
        print("Storing resulting {} image at {}".format(self.image_name, store_path))
        # store_path = "datasets/Fluo-N2DH-SIM+"
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
        seg_path = store_path + "/mask{}.tif".format(self.image_name)
        cv2.imwrite(seg_path, self.seg_output.astype(np.uint16))

    def plot(self):
        size = 2 + [isinstance(self.binary_seg, type(None)), isinstance(self.seg_output, type(None))].count(False)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, size, 1)
        plt.imshow(self.image)
        plt.title("Input image")

        if not isinstance(self.seg_gt, type(None)):
            plt.subplot(1, size, 2)
            plt.imshow(self.seg_gt)
            plt.title("Labelled")

        elif not isinstance(self.seg_output, type(None)):
            plt.subplot(1, size, 2)
            plt.imshow(self.seg_output)
            plt.title("Labelled")

        if not isinstance(self.binary_seg, type(None)):
            plt.subplot(1, size, size)
            plt.imshow(self.binary_seg)
            plt.title("Ours")
        plt.show()

    def __str__(self):
        return "seq: {}, frame: {}, num of cells: {}".format(self.sequence, self.image_name, self.cell_count)

    def __repr__(self):
        return "seq: {}, frame: {}, num of cells: {}".format(self.sequence, self.image_name, self.cell_count)


class CellTracker:
    def __init__(self, siamese_model_path, unet_path, dataset_path, use_cuda, new_w, new_h):
        self.dataset_path = dataset_path
        self.unet_path = unet_path
        self.new_w = new_w
        self.new_h = new_h

        self.tracker = TrackerSiamFC(net_path=siamese_model_path, use_cuda=use_cuda)
        if unet_path is not None:
            print("Loading pretrained model")
            self.seg_net, pretrained = self.load_unet(), True
        else:
            print("Did not load pretrained model")
            self.seg_net, pretrained = None, False

        self.train_images = None
        self.tracks = {}
        self.track_count = 0
        self.result = []
        self.set_01, self.set_02 = None, None

    def load_unet(self):
        model = create_model(self.unet_path, self.new_w, self.new_h)
        return model

    def load_evaluation_images(self, sequence, extension=".tif"):
        seg_dir = "/0{}".format(sequence)
        result = []
        print("Loading test images from {}".format(os.path.join(self.dataset_path + seg_dir, "*" + extension)))
        for frame_id, img_path in enumerate(
                glob.glob(os.path.join(self.dataset_path + seg_dir, "*" + extension))):
            # name = img_path.split("\\t")[-1].split(extension)[0]
            name = img_path.split("/t")[-1].split(extension)[0]
            # print("Image name: {}".format(name))
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            seg_img = None
            result.append(CellImage(img, name, self.dataset_path, sequence, seg_img))
        result = sorted(result, key=lambda x: x.image_name, reverse=False)
        return result

    # predicts binary segmentation for input image using the unet
    def predict_seg(self, input_img, thr_markers=240, thr_cell_mask=230):

        w = np.shape(input_img)[0]
        h = np.shape(input_img)[1]
        img = cv2.equalizeHist(np.minimum(input_img, 255).astype(np.uint8)) / 255
        img = img.reshape((1, w, h, 1)) - .5

        if self.new_w > 0 or self.new_h > 0:
            img2 = np.zeros((1, self.new_w, self.new_h, 1), dtype=np.float32)
            img2[:, :w, :h, :] = img
            img = img2

        prediction = self.seg_net.predict(img, batch_size=1)
        # prediction = prediction[0, :w, :h, 1]

        # New watershed
        # naive_seg = postprocess_cell_mask(prediction[0, :w, :h, 3] * 255, threshold=thr_cell_mask)
        # distance = ndi.distance_transform_edt(naive_seg)
        # local_maxi = peak_local_max(distance, labels=naive_seg, footprint=np.ones((15, 15)), indices=False)
        # markers = ndi.label(local_maxi)[0]
        # prediction = watershed(-distance, markers, mask=naive_seg)

        # # Watershed
        # m = prediction[0, :w, :h, 1] * 255
        # c = prediction[0, :w, :h, 3] * 255
        # o = (img + .5) * 255
        #
        # # postprocess the result of prediction
        # idx, markers = postprocess_markers(m, threshold=thr_markers, erosion_size=1, circular=False,
        #                                    step=30)
        # cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)
        # # correct border
        # cell_mask = np.maximum(cell_mask, markers)
        # prediction = (watershed(-c, markers, mask=cell_mask) > 0)*1.0

        # # Previous unet
        # img = input_img / 255
        # img = torch.Tensor(list(transform.resize(input_img, (512, 512), mode='symmetric'))).unsqueeze(0).unsqueeze(
        #     0).permute(0, 2, 3, 1).numpy()
        # prediction = self.seg_net.predict(img)
        #
        # prediction = cv2.resize(prediction[0, :, :, 0], tuple(reversed(input_img.shape)))
        # prediction = np.array(prediction)
        # _, prediction = cv2.threshold(prediction, 0.6, 1, cv2.THRESH_BINARY)
        # prediction = prediction#.astype(np.uint16)

        return prediction

    # predict segmentation for all frames:
    # def segment_images(self, sequence):
    #     for cell_img in tqdm(self.train_images[sequence]):
    #         cell_img.binary_seg = self.predict_seg(cell_img.image)

    # writes the frames with cell locations to a video file
    def store_footage(self, sequence, fps: int = 3):

        # output_data = np.array([x for x in self.result])
        # output_data = output_data.astype(np.uint8)
        # skvideo.io.vwrite("result {} 0{}.mp4".format(self.name, sequence), output_data, inputdict={'-r': str(fps)})

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/result_0{}.avi".format(self.dataset_path, sequence), fourcc, float(fps),
                              (self.result[0].shape[1], self.result[0].shape[0]))
        for frame in self.result:
            out.write(frame.astype(np.uint8))
        out.release()

    # predict the new location of a cell located in frame1:
    def predict_cell_location(self, frame1, frame2, cell):
        self.tracker.init(frame1, [cell.min_col, cell.min_row, cell.width, cell.height])
        [x, y, w, h] = self.tracker.update(frame2)
        return [int(x), int(y), int(x + w), int(y + h)]

    # uses random walker with markers from previous frame to predict a new segmentation
    # for collided cells
    @staticmethod
    def resegmentation(initial_segmentation, local_maxi):
        markers = measure.label(local_maxi)
        markers[~initial_segmentation] = -1
        labels = random_walker(initial_segmentation, markers)
        labels[labels == -1] = 0
        return labels

    # stores the tracks' desciptions in the right format
    def store_track(self, filename, sequence):
        store_path = os.path.join(self.dataset_path, "0{}_RES/".format(sequence), filename)
        keys = sorted(list(self.tracks.keys()))
        print("Storing track at {}".format(store_path))
        with open(store_path, 'w', encoding='utf-8') as file:
            for k in keys:
                file.write(str(self.tracks[k]) + "\n")

    # get the backward matches and resegment when a collision is detected:
    def get_new_detections_dict(self, previous_frame, prev_img, current_frame, track_dict, alt=True):

        cur_img = np.stack((current_frame.image.astype(np.int16),) * 3, axis=-1)

        available_cells = current_frame.get_cell_locations()
        # print("available_cells: {}".format(available_cells))
        new_detections_dict = {c: [] for c in available_cells}

        prev_num_cells = len(new_detections_dict)
        cur_num_cells = 0
        # return current_frame.seg_output, new_detections_dict, cur_img

        while alt and prev_num_cells != cur_num_cells:
            prev_num_cells = cur_num_cells
            alt = False

            # cell detection backward pass
            for dest_cell in new_detections_dict.keys():

                # get the predicted location of the cell in the previous frame
                [tl_x, tl_y, br_x, br_y] = self.predict_cell_location(cur_img, prev_img, deepcopy(dest_cell))

                # match all cell located in that area
                for cell_track in track_dict.keys():
                    c = cell_track.current_cell

                    if tl_x < c.centroid_x < br_x and tl_y < c.centroid_y < br_y:
                        try:
                            new_detections_dict[dest_cell].append(cell_track)
                        except Exception as e:
                            print("new_detections_dict.keys(): {}".format(new_detections_dict.keys()))
                            print("dest_cell: {}".format(dest_cell))
                            # print(new_detections_dict[str(dest_cell)])
                            print(new_detections_dict[dest_cell])
                            print("\n\n")
                            print(new_detections_dict)
                            raise e

                # forward checking:
                if len(new_detections_dict[dest_cell]) > 1:
                    final = []
                    for c_t in new_detections_dict[dest_cell]:
                        [tl_x, tl_y, br_x, br_y] = self.predict_cell_location(prev_img, cur_img, c_t.current_cell)
                        pred_center_x, pred_center_y = int((br_x + tl_x) / 2), int((br_y + tl_y) / 2)

                        if tl_x < dest_cell.centroid_x < br_x and tl_y < dest_cell.centroid_y < br_y:
                            final.append(c_t)

                        elif dest_cell.min_col < pred_center_x < dest_cell.max_col and \
                                dest_cell.min_row < pred_center_y < dest_cell.max_row:
                            final.append(c_t)
                    new_detections_dict[dest_cell] = final

                # if two or more cells are matching a collision has occured and the frame has to be resegmented
                if len(new_detections_dict[dest_cell]) > 1:
                    alt = True
                    tl_x = min([t.current_cell.min_col for t in new_detections_dict[dest_cell]])
                    tl_y = min([t.current_cell.min_row for t in new_detections_dict[dest_cell]])
                    br_x = max([t.current_cell.max_col for t in new_detections_dict[dest_cell]])
                    br_y = max([t.current_cell.max_row for t in new_detections_dict[dest_cell]])

                    frame1_seg = previous_frame.seg_output[tl_y:br_y, tl_x:br_x]
                    distance = np.zeros((int(frame1_seg.shape[0]), int(frame1_seg.shape[1])))

                    for color, t in enumerate(new_detections_dict[dest_cell]):
                        a = max(0, int(t.current_cell.centroid_y) - tl_y - 1)
                        b = max(0, int(t.current_cell.centroid_x) - tl_x - 1)
                        distance[a:a + 2, b:b + 2] = np.array(np.full((2, 2), color + 1))

                    distance = cv2.resize(distance, (int(dest_cell.width), int(dest_cell.height)),
                                          interpolation=cv2.INTER_NEAREST)

                    current_frame.seg_output[current_frame.seg_output == dest_cell.color] = 0
                    new_seg = self.resegmentation(dest_cell.segmentation, distance)

                    # plot new segmentation result:
                    # self.multiplot([frame1_seg, dest_cell.segmentation, new_seg])
                    global bla_count
                    bla_count += 1
                    # print(bla_count)
                    # vis = np.concatenate((dest_cell.segmentation, new_seg), axis=1)
                    # plt.imshow(vis)
                    # plt.show()

                    current_frame.seg_output[dest_cell.min_row:dest_cell.max_row,
                    dest_cell.min_col:dest_cell.max_col] = np.maximum(
                        current_frame.seg_output[dest_cell.min_row:dest_cell.max_row,
                        dest_cell.min_col:dest_cell.max_col], new_seg)

                    # relabel the frame and redetect the cells
                    current_frame.seg_output = measure.label(current_frame.seg_output)
                    available_cells = current_frame.get_cell_locations()
                    new_detections_dict = {c: [] for c in available_cells}
                    cur_num_cells = len(new_detections_dict)
                    break

                    # display frame final segmentation:
                    # plt.imshow(current_frame.seg_output)
                    # plt.show()
        return current_frame.seg_output, new_detections_dict, cur_img

    @staticmethod
    def multiplot(image_list):
        abc = ["A", "B", "C", "D", "E", "F", "G", "H"]
        # abc2 = ["t=i", "t=i+1", "t=i+1", "t=i+1"]
        fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=(10, 5))

        for num, x in enumerate(image_list):
            plt.subplot(1, len(image_list), num + 1)
            plt.title(abc[num], fontsize=25)
            plt.axis('off')
            plt.imshow(x)
        # plt.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
        # fig.tight_layout()
        plt.savefig("segres.png", bbox_inches='tight')


    @staticmethod
    # makes sure that all cells in a given track have the same pixel value in segmentation
    def propagate_labels(frame, living_tracks, track, cell):
        free_label = max(np.unique(frame.seg_output)) + 1

        if track.cell_id != cell.color:
            for swap_id, swap_track in living_tracks:
                if swap_id != track.cell_id and swap_track.current_cell.color == track.cell_id:
                    frame.seg_output = np.where(frame.seg_output == swap_track.current_cell.color, free_label,
                                                frame.seg_output)
                    swap_track.current_cell.color = free_label
                    break

            frame.seg_output = np.where(frame.seg_output == cell.color, track.cell_id, frame.seg_output)
            cell.color = track.cell_id
        return frame, living_tracks

    # runs the tracking algorithm and exports results for evaluation
    def run_test(self, sequence, collision_detection=True, store_footage=0, load_segs_from_file=None):
        set_01 = self.load_evaluation_images(sequence)
        self.result = []

        print("Segmenting footage:")
        # apply initial segmentation to footage:
        if load_segs_from_file is None:
            for frame in tqdm(set_01, position=0):
                frame.binary_seg = self.predict_seg(frame.image)
                frame.binary_seg_to_output()
        else:
            imgs_list = sorted(make_list_of_imgs_only(os.listdir(load_segs_from_file), 'tif'), key=natural_keys)
            for frame in tqdm(set_01, position=0):
                img_indx = int(frame.image_name)
                img_path = os.path.join(load_segs_from_file, imgs_list[img_indx])
                print("reading img: {} at {}".format(img_indx, img_path))
                frame.binary_seg = mpimg.imread(img_path)
                frame.binary_seg_to_output()

        # load first frame
        previous_frame = set_01[0]
        prev_img = np.stack((previous_frame.image.astype(np.int16),) * 3, axis=-1)

        # init tracks from detected cells in first frame
        self.tracks = {c_id + 1: CellTrack(c, 0, c_id + 1, 0) for c_id, c in
                       enumerate(previous_frame.get_cell_locations())}
        self.track_count = len(self.tracks) + 1

        i = 0
        print("Tracking:")
        for current_frame in tqdm(set_01[1:]):
            track_dict = {t: [] for t in self.tracks.values() if t.alive}

            # solve collisions in segmentation and locate all cells in the new frame:
            current_frame.seg_output, new_detections_dict, cur_img = \
                self.get_new_detections_dict(previous_frame, prev_img, current_frame, track_dict, collision_detection)

            # image for video visualisation
            cur_copy = cur_img.copy()

            # cell detection forward pass:
            for cell_track in track_dict.keys():
                cell = cell_track.current_cell

                [tl_x, tl_y, br_x, br_y] = self.predict_cell_location(prev_img, cur_img, cell)
                pred_center_x, pred_center_y = int((br_x + tl_x) / 2), int((br_y + tl_y) / 2)

                for c in new_detections_dict.keys():
                    if c.min_col < pred_center_x < c.max_col and c.min_row < pred_center_y < c.max_row:
                        track_dict[cell_track].append(c)
                    elif tl_x < c.centroid_x < br_x and tl_y < c.centroid_y < br_y:
                        track_dict[cell_track].append(c)

            # match cells from the previous frame to newly located cells:
            new_tracks = {}
            for track_id, cell_track in self.tracks.items():
                matched = False

                # if death cell add it to the new stack
                if not cell_track.alive:
                    new_tracks[track_id] = cell_track

                # else try to find a track continuation
                elif cell_track.alive:

                    forward_match = track_dict[cell_track]

                    # case 1->_
                    if not forward_match:

                        # check if 1<-1:
                        for dest_cell, t in new_detections_dict.items():
                            if cell_track in t:
                                cell_track.add_cell(dest_cell)
                                matched = True
                                break

                        # remove the cell from free new detection
                        if matched:
                            del new_detections_dict[cell_track.current_cell]

                        # else the cell has no match and has died:
                        else:
                            cell_track.alive = False
                        new_tracks[track_id] = cell_track

                    # case 1->1
                    elif len(forward_match) == 1:
                        dest_cell = forward_match[0]

                        # should not occur:
                        if dest_cell not in new_detections_dict:
                            cell_track.alive = False

                        # if 1<-1 or _<-1 match
                        elif not new_detections_dict[dest_cell] or cell_track in new_detections_dict[dest_cell]:
                            cell_track.add_cell(dest_cell)
                            del new_detections_dict[dest_cell]

                        # if 2<-1 death
                        else:
                            cell_track.alive = False
                        new_tracks[track_id] = cell_track

                    # case 1->1,2 (mitosis)
                    else:
                        available = [c for c in forward_match if c in new_detections_dict]

                        if not available or len(available) > 1:
                            cell_track.alive = False
                            for dest_cell in available:
                                del new_detections_dict[dest_cell]
                                new_track = CellTrack(dest_cell, i + 1, self.track_count, cell_track.cell_id)
                                new_tracks[self.track_count] = new_track
                                self.track_count += 1
                        else:
                            del new_detections_dict[available[0]]
                            cell_track.add_cell(available[0])

                        new_tracks[track_id] = cell_track

            # create new tracks for the unmatched cells in the new frame:
            for dest_cell, t in new_detections_dict.items():
                new_track = CellTrack(dest_cell, i + 1, self.track_count, 0)
                new_tracks[self.track_count] = new_track
                self.track_count += 1

            living_tracks = [(track_id, track) for track_id, track in new_tracks.items() if track.last_frame == i + 1]
            # displaying cell locations on frame for visual evaluation:
            for track_id, track in living_tracks:
                c = track.current_cell
                current_frame, living_tracks = self.propagate_labels(current_frame, living_tracks, track, c)
                cv2.rectangle(cur_copy, c.tl, c.br, track.display_color, 3)
                x, y = int(c.centroid_x), int(c.centroid_y)
                cv2.putText(cur_copy, str(track_id), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, track.display_color, 1, cv2.LINE_AA)
            cv2.putText(cur_copy, str(i + 1), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1, cv2.LINE_AA)

            self.result.append(cur_copy)
            self.tracks = new_tracks
            previous_frame = current_frame
            prev_img = cur_img
            i += 1

        print("Saving results")
        for frame in tqdm(set_01):
            frame.store()
        self.store_track("res_track.txt", sequence)

        if store_footage:
            print("Creating video")
            self.store_footage(sequence=sequence, fps=3)
