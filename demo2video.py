import numpy as np
import os
import torch
import glob
import cv2
import pdb

image_dir = "F:\\AI28_20201130\\output\\frost"

images_dir = glob.glob(os.path.join(image_dir,"*.jpg"))

image_len = len(images_dir)

h,w,_ = cv2.imread(images_dir[0]).shape

out = cv2.VideoWriter(image_dir+"\\"+'output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, (w,h))

for i in range(image_len):
    img = cv2.imread(os.path.join(image_dir,"%d.jpg"%i))
    # pdb.set_trace()
    out.write(img)

out.release()