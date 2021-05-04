import glob
import os
import tifffile
import numpy as np

data_folder = ["/home/jb/Desktop/Data_single_loci/Simulation_2021_05_04/Data_1/",
               "/home/jb/Desktop/Data_single_loci/Simulation_2021_05_04/Data_2/"]
dest_folder = "/home/jb/Desktop/Data_single_loci/Simulation_2021_05_04/"

data_type = "Deconvolved" # "Raw"

for folder in data_folder:

    raw_data = []
    gt_data = []

    # List all the simulated data (either deconvolved or raw) and the associated ground-truth
    # ---------------------------------------------------------------------------------------

    if data_type == "Deconvolved":
        os.chdir(folder + 'Deconvolved_data')
        raw_data = glob.glob('ROI_*_converted_decon.tif')
    else:
        os.chdir(folder + 'Raw_data')
        raw_data = glob.glob('ROI_*.tif')

    for n in range(len(raw_data)):
        raw_data[n] = os.path.abspath(raw_data[n])

    os.chdir(folder + 'GT')
    gt_data = glob.glob('GT_ROI_*.tif')

    for n in range(len(gt_data)):
        gt_data[n] = os.path.abspath(gt_data[n])

    # For each movie, create a stack of image with 500nm separation between each plane
    # --------------------------------------------------------------------------------

    for n_file in range(len(raw_data)):

        raw = tifffile.imread(raw_data[n_file])
        gt = tifffile.imread(gt_data[n_file])

        raw_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/5)))
        gt_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/5)))

        n_frame = round(raw.shape[1]/5)
        for frame in range(n_frame):
            raw_small[:, :, frame] = raw[0, 5*frame, :, :]
            gt_small[:, :, frame] = gt[5 * frame, :, :]

    # Save the movies, either for the training or the testing set
    # -----------------------------------------------------------

        p = np.random.binomial(1, 0.85)

        if p == 1:
            os.chdir(dest_folder + 'Training/raw')
        else:
            os.chdir(dest_folder + 'Test/raw')

        name = str(len(glob.glob('*.tif'))) + "_raw.tif"
        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = raw_small[:, :, n]
                tf.save(im.astype(np.uint16))

        if p == 1:
            os.chdir(dest_folder + 'Training/label')
        else:
            os.chdir(dest_folder + 'Test/label')

        name = str(len(glob.glob('*.tif'))) + "_label.tif"
        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = gt_small[:, :, n]
                tf.save(im.astype(np.uint16))
