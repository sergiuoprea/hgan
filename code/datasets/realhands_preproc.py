import os
import argparse
import logging
from time import sleep
import numpy as np
import cv2 as cv
import tqdm as tqdm
import multiprocessing as mp

HELP_MSG = """
The dataset can be downloaded from https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm
The dataset provide hand segmentation masks, they present artifacts such as speckle noise, gaps in the
hand masks and they are not binary. With this code we aim to better segment the hands. New masks will be
saved alongside the provided one (*_mask.png).

Example usage:

python -O prepare_realhands_dataset.py
"""

def get_mask(path, fname):
    """
    Firstly we get a raw hand segmentation mask, by removing the green chroma. We will postprocess
    these masks as there are still artifacts and they present random noise. We proceed with a 
    median blur smoothing to remove random speckle noise. After that we apply Otsu tresholding to
    binarize the mask. In order to remove artifacts and small pixel clusters, we search fo contours 
    to only consider the biggest one according to they area. To fill some gaps in the hands we rely
    on morphological filters, i.e. dilation and erosion.
    """
    basename = os.path.basename(fname)
    mask_path = os.path.join(path, "mask",
                             str(os.path.splitext(basename)[0].split('_c')[0]) + "_mask.jpg")

    # 1. Get a raw hand segmentation
    mask = remove_green_chroma(fname)
    
    # 2. Filter the masks according to the hand presence. There are some frames
    # where only a small part of the hand is visible. We avoid these frames
    # thresholding by the number of pixels representing the hand.
    tot_num_pixels = (mask.shape[0] * mask.shape[1])
    min_num_pixels = tot_num_pixels * 0.05
    max_num_pixels = tot_num_pixels * 0.50
    num_mask_pixels = np.count_nonzero(mask)

    if num_mask_pixels > max_num_pixels or num_mask_pixels < min_num_pixels:
        logging.info(fname)
        return

    # 3. To grayscale
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY) 

    # 4. Applying a subtle median blurring to remove random speckle noise
    blur = cv.medianBlur(mask, 3)

    # 5. Otsu thresholding to binarize the image
    otsu_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # 6. Searching for the existing contours in the mask.
    # - In a perfect mask a single contour should be detected representing the hand
    contours = cv.findContours(otsu_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # 7. Just remove images with more than 60 contours fact that denotes artifact presence.
    # The threshold was experimentally established.
    if len(contours) > 60 or len(contours) == 0:
        logging.info(fname)
        return

    max_contour = 0
    for i in range(1, len(contours)):
        if cv.contourArea(contours[max_contour]) > cv.contourArea(contours[i]):
            cv.drawContours(otsu_thresh, [contours[i]], -1, (0, 0, 0), -1)
        else:
            cv.drawContours(otsu_thresh, [contours[max_contour]], -1, (0, 0, 0), -1)
            max_contour = i

    # 8. Morph close operation
    closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closing = cv.morphologyEx(otsu_thresh, cv.MORPH_CLOSE, closing_kernel, iterations=1)

    # 9. Subtle erosion operation to remove some green pixels in the hand border
    erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    output = cv.erode(closing, erode_kernel, iterations=1)

    if __debug__:
        print("Processed image %s" % path)
        cv.imwrite('blured_output.png', blur)
        cv.imwrite('otsu_threshold_output.png', otsu_thresh)
        cv.imwrite('output.png', output)

    cv.imwrite(mask_path, output)


def remove_green_chroma(img_path):
    """
    Function used to segment the green chroma. Hands will be represented in white while
    the background will be black.
    """
    img = cv.imread(img_path)
    mask = np.zeros_like(img)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    chroma = cv.inRange(img_hsv, (30, 30, 40), (90, 255, 255))
    mask[chroma == 0] = 255 # hands are represented by white pixels

    return mask

def process_folder(path):
    """
    Function used to process the segmentation masks of a given folder given its path.
    """

    print("Processing folder %s ..." %(path))

    for dirpath, _, files in os.walk(os.path.join(path, "color")):
        for filename in files:
            fname = os.path.join(dirpath, filename)

            if fname.endswith((".jpg", ".png")) and not fname.endswith("_fixed.png"):
                get_mask(path, fname)

def process_dataset(path):
    """
    Function used to preprocess the masks of RealHands dataset.
    """
    process_folder(path)

if __name__ == '__main__':

    logging.basicConfig(filename='broken_masks.log', filemode='w',
                        level=logging.INFO, format='%(message)s')

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root folder of the dataset.')

    OPT = PARSER.parse_args()

    print(HELP_MSG)

    # Searching for all the mask directories in the dataset
    print("Searching for all the user recordings existing in the dataset...")
    USER_RECORDINGS = [os.path.join(OPT.data_dir, recording) 
                       for recording in sorted(os.listdir(OPT.data_dir))
                       if os.path.isdir(os.path.join(OPT.data_dir, recording))]

    print(USER_RECORDINGS)
    print("Using %d cores to preprocess the data." % mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    pool.map(process_dataset, USER_RECORDINGS)