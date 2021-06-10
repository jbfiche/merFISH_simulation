import glob
import os
import tifffile
import numpy as np
import shutil
import re

np.random.seed(2)

main_folder = "/home/jb/Desktop/Data_single_loci/3D_simulations/2021-06-10_14-50"
data_folder = ["/home/jb/Desktop/Data_single_loci/3D_simulations/2021-06-10_14-50/Raw",
               "/home/jb/Desktop/Data_single_loci/3D_simulations/2021-06-10_14-50/Deconvolved"]
gt_data_folder = "/home/jb/Desktop/Data_single_loci/3D_simulations/2021-06-10_14-50/Threshold_2"

# According to the type of data we want to use, define the folder where the
# training set will be saved
# -------------------------- 

data_types = ["Raw", "Deconvolved"]
for n_type in range(len(data_types)):
    
    training_folder_name = "Training_data_" + data_types[n_type]
    training_folder_path = os.path.join(main_folder, training_folder_name)
    if os.path.isdir(training_folder_path):
        shutil.rmtree(training_folder_path)
    
    os.mkdir(training_folder_path)

    # Define the folders for the training and testing data
    # ----------------------------------------------------
    
    os.chdir(training_folder_path)
    if os.path.isdir('train'):
        shutil.rmtree('train')
        shutil.rmtree('test')
    
    os.mkdir('train')
    os.mkdir('train/raw')
    os.mkdir('train/label')
    
    os.mkdir('test')
    os.mkdir('test/raw')
    os.mkdir('test/label')
    
    
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
    
    
    os.chdir(data_folder[n_type])
    raw_data = glob.glob('ROI_*.tif')
    raw_data.sort(key=natural_keys)
    print(raw_data)

    for n in range(len(raw_data)):
        raw_data[n] = os.path.abspath(raw_data[n])

    os.chdir(gt_data_folder)
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

        if data_types[n_type] == "Deconvolved":
            raw_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/2)))
            gt_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/2)))

            n_frame = round(raw.shape[1] / 2)
            for frame in range(n_frame):
                raw_small[:, :, frame] = raw[0, 2 * frame, :, :]
                gt_small[:, :, frame] = gt[2 * frame, :, :]

        elif data_types[n_type] == "Raw":

            raw_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/2)))
            gt_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/2)))

            n_frame = round(raw.shape[0]/2)
            for frame in range(n_frame):
                raw_small[:, :, frame] = raw[2 * frame, :, :]
                gt_small[:, :, frame] = gt[2 * frame, :, :]

    # Save the movies, either for the training or the testing set
    # -----------------------------------------------------------

        p = np.random.binomial(1, 0.85)
        
        if p == 1:
            saving_path = os.path.join(training_folder_path, 'train', 'raw')
        else:
            saving_path = os.path.join(training_folder_path, 'test', 'raw')
            
        os.chdir(saving_path)
        name = str(len(glob.glob('*.tif'))) + "_raw.tif"

        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = raw_small[:, :, n]
                tf.save(im.astype(np.uint16))
        
        if p == 1:
            saving_path = os.path.join(training_folder_path, 'train', 'label')
        else:
            saving_path = os.path.join(training_folder_path, 'test', 'label')
            
        os.chdir(saving_path)
        name = str(len(glob.glob('*.tif'))) + "_label.tif"

        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = gt_small[:, :, n]
                tf.save(im.astype(np.uint16))
