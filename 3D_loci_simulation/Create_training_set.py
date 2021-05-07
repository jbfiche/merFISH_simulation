import glob
import os
import tifffile
import numpy as np
import shutil
import re

data_folder = ["/home/jb/Desktop/Data_single_loci/Simulation_06_05_21/"]
dest_folder = "/home/jb/Desktop/Data_single_loci/Simulation_06_05_21/Training_data_raw/"

data_type = "Raw" #"Deconvolved" #

# Define the folders for the training and testing data
# ----------------------------------------------------

os.chdir(dest_folder)
if os.path.isdir('train'):
    shutil.rmtree('train')
    shutil.rmtree('test')

os.mkdir('train')
os.mkdir('train/raw')
os.mkdir('train/label')

os.mkdir('test')
os.mkdir('test/raw')
os.mkdir('test/label')

for folder in data_folder:

    raw_data = []
    gt_data = []

    # List all the simulated data (either deconvolved or raw) and the associated ground-truth
    # ---------------------------------------------------------------------------------------

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    if data_type == "Deconvolved":
        os.chdir(folder + 'Deconvolved')
        raw_data = glob.glob('ROI_*_converted_decon.tif')
    else:
        os.chdir(folder + 'Raw')
        raw_data = glob.glob('ROI_*.tif')

    raw_data.sort(key=natural_keys)
    print(raw_data)

    for n in range(len(raw_data)):
        raw_data[n] = os.path.abspath(raw_data[n])

    os.chdir(folder + 'GT')
    gt_data = glob.glob('GT_ROI_*.tif')

    gt_data.sort(key=natural_keys)
    print(gt_data)

    for n in range(len(gt_data)):
        gt_data[n] = os.path.abspath(gt_data[n])

    # For each movie, create a stack of image with 500nm separation between each plane
    # --------------------------------------------------------------------------------

    for n_file in range(len(raw_data)):

        raw = tifffile.imread(raw_data[n_file])
        gt = tifffile.imread(gt_data[n_file])

        if data_type == "Deconvolved":
            raw_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/5)))
            gt_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/5)))

            n_frame = round(raw.shape[1] / 5)
            for frame in range(n_frame):
                raw_small[:, :, frame] = raw[0, 5 * frame, :, :]
                gt_small[:, :, frame] = gt[5 * frame, :, :]

        elif data_type == "Raw":

            raw_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/5)))
            gt_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/5)))

            n_frame = round(raw.shape[0]/5)
            for frame in range(n_frame):
                raw_small[:, :, frame] = raw[5*frame, :, :]
                gt_small[:, :, frame] = gt[5 * frame, :, :]

    # Save the movies, either for the training or the testing set
    # -----------------------------------------------------------

        p = np.random.binomial(1, 0.85)

        if p == 1:
            os.chdir(dest_folder + 'train/raw')
        else:
            os.chdir(dest_folder + 'test/raw')

        name = str(len(glob.glob('*.tif'))) + "_raw.tif"
        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = raw_small[:, :, n]
                tf.save(im.astype(np.uint16))

        if p == 1:
            os.chdir(dest_folder + 'train/label')
        else:
            os.chdir(dest_folder + 'test/label')

        name = str(len(glob.glob('*.tif'))) + "_label.tif"
        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = gt_small[:, :, n]
                tf.save(im.astype(np.uint16))
