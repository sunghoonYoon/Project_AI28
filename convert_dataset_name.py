import glob
import os
import cv2


dataset_dir = "F:\\AI28_20201006"

folders = glob.glob(os.path.join(dataset_dir,"*"))

save_root_dir = "F:\\AI_dataset3"

print(folders)

total_files=0

for folder in folders:
    sub_folders = glob.glob(os.path.join(folder,"*"))
    for sub_folder in sub_folders:

        ab_type = os.path.basename(folder)
        files = glob.glob(os.path.join(sub_folder,"*.png"))

        for file in files:

            img = cv2.imread(file)
            name = ab_type+"_" +"%05d" % total_files+".png"
            total_files +=1

            save_dir = os.path.join(save_root_dir,name)

            cv2.imwrite(save_dir,img)

        # total_files+=len(files)

print(total_files)