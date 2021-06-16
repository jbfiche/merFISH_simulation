import glob
import os
import tifffile
import numpy as np
import shutil
import re

np.random.seed(2)

main_folder = "/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-16_10-51"
data_folder = ["/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-16_10-51/Raw",
               "/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-16_10-51/Deconvolved"]
gt_data_folder = "/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-16_10-51/Threshold_4"

# According to the type of data we want to use, define the folder where the
# training set will be saved
# -------------------------- 

data_types = ["Raw", "Deconvolved"]
for n_type in range(len(data_types)):
    
    training_folder_name = "Training_data_thresh_4_" + data_types[n_type]
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
        shutil.rmtree('mask')
    
    os.mkdir('train')
    os.mkdir('train/raw')
    os.mkdir('train/label')
    os.mkdir('train/mask')
    
    os.mkdir('test')
    os.mkdir('test/raw')
    os.mkdir('test/label')
    os.mkdir('test/mask')
    
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
    mask_data = glob.glob('mask_ROI_*.tif')
    gt_data.sort(key=natural_keys)
    mask_data.sort(key=natural_keys)
    print(gt_data)
    print(mask_data)

    for n in range(len(gt_data)):
        gt_data[n] = os.path.abspath(gt_data[n])
        mask_data[n] = os.path.abspath(mask_data[n])
        

    # For each movie, create a stack of image with 500nm separation between each plane
    # --------------------------------------------------------------------------------

    for n_file in range(len(raw_data)):

        raw = tifffile.imread(raw_data[n_file])
        gt = tifffile.imread(gt_data[n_file])
        mask = tifffile.imread(mask_data[n_file])

        if data_types[n_type] == "Deconvolved":
            raw_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/2)))
            gt_small = np.zeros((raw.shape[2], raw.shape[3], round(raw.shape[1]/2)))

            n_frame = round(raw.shape[1] / 2)
            for frame in range(n_frame):
                av = (raw[0, 2 * frame, :, :].astype(np.float) + raw[0, 2 * frame + 1, :, :].astype(np.float))/2
                raw_small[:, :, frame]  = av.astype(np.uint16)
                gt_small[:, :, frame] = np.maximum(gt[2 * frame, :, :],gt[2 * frame + 1, :, :])

        elif data_types[n_type] == "Raw":
            raw_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/2)))
            gt_small = np.zeros((raw.shape[1], raw.shape[2], round(raw.shape[0]/2)))

            n_frame = round(raw.shape[0]/2)
            for frame in range(n_frame):
                av = (raw[2 * frame, :, :].astype(np.float) + raw[2 * frame + 1, :, :].astype(np.float))/2
                raw_small[:, :, frame]  = av.astype(np.uint16)
                gt_small[:, :, frame] = np.maximum(gt[2 * frame, :, :], gt[2 * frame + 1, :, :])

    # Save the movies, either for the training or the testing set
    # -----------------------------------------------------------

        p = np.random.binomial(1, 0.9)
        
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
            saving_path_mask = os.path.join(training_folder_path, 'train', 'mask')
            
        else:
            saving_path = os.path.join(training_folder_path, 'test', 'label')
            saving_path_mask = os.path.join(training_folder_path, 'test', 'mask')
            
        os.chdir(saving_path)
        name = str(len(glob.glob('*.tif'))) + "_label.tif"

        with tifffile.TiffWriter(name) as tf:
            for n in range(n_frame):
                im = gt_small[:, :, n]
                tf.save(im.astype(np.uint16))
            
        os.chdir(saving_path_mask)
        name = str(len(glob.glob('*.tif'))) + "_mask.tif"

        with tifffile.TiffWriter(name) as tf:
            tf.save(mask.astype(np.uint16))
