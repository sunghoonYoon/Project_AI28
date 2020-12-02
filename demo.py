import glob
import os
import cv2
import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
import pdb
import torch.nn.functional as F
# from torchvision.models.resnet import resnet101, resnet50
from PIL import Image
from network.resnet import resnet50

from ai_dataset import AI_Dataset
import matplotlib.pyplot as plt


def returnCAM(feature_conv, weight_softmax, img, h_x,labels):
    # im_h, im_w = img.shape[1], img.shape[0]
    im_h, im_w = img.size()[2],img.size()[3]
    # print(im_h,im_w)
    probs, idx = h_x.sort(0, True)


    size_upsample = (im_w, im_h)  #####BE CAREFUL
    batch, nc, h, w = feature_conv.shape
    output_cam = []

    cams = torch.zeros((batch, 21, h,w))
    # fgs = torch.zeros((batch,1,im_h,im_w))

    for i in range(batch):
        # label_idxs = list(np.nonzero(labels[i]))

        cam_dict = {}
        # for j in label_idxs:
        for j in range(2):
            # j = j.item()
            cam = torch.mm(weight_softmax[j].clone().unsqueeze(0), (feature_conv[i].reshape((nc, h * w))))
            cam = cam.reshape(h, w)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            # cam_mean= torch.mean(cam_img)
            # cam = (255 * cam).float()
            # print(cam_img.shape)
            # if cam_mean<0.33:
            # cams[i,j+1, :, :] = torch.from_numpy(cv2.resize(cam.cpu().detach().numpy(),size_upsample))
            # cam = F.upsample(cam.unsqueeze(0), size_upsample, mode='bilinear', align_corners=False)[0]
            # cam = cam - torch.min(cam)
            # cam = cam / torch.max(cam)
            cams[i, j, :, :] = cam#*labels[j]
            cam_dict[j] =  cv2.resize(cam.cpu().detach().numpy(),size_upsample)

    cams = F.upsample(cams,(im_h,im_w),mode='bilinear',align_corners=False)


    # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cams

mean_img = [0.485, 0.456, 0.406]
std_img = [0.229, 0.224, 0.225]


inv_normalize = transforms.Normalize(
    mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
    std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
)

data_dir = "F:\\AI28_20201130\\sequence\\frost\\0"
save_dir = "F:\\AI28_20201130\\output"

type_dir = os.path.basename(os.path.dirname(data_dir))

images = glob.glob(os.path.join(data_dir,"*.png"))

resnet = resnet50(pretrained=True).cuda()

if not os.path.isdir(os.path.join(save_dir,type_dir)):
    os.mkdir(os.path.join(save_dir,type_dir))


# state_dict = torch.load("C:\\Users\\sunghoon Yoon\\PycharmProjects\\AI28\\model\\model_best_resnet_v3.pth")
state_dict = torch.load("C:\\Users\\sunghoon Yoon\\PycharmProjects\\AI28\\model\\model_resnet_er_v3.pth")
resnet.load_state_dict(state_dict)

resnet.eval()

for i in range(len(images)):
    image = Image.open(os.path.join(data_dir,"%d"%i+"_0.png")).convert("RGB")
    image = np.float64(image)/255.0
    image = image-np.array([0.485, 0.456, 0.406])
    image = image/np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).type(torch.FloatTensor)

    # pdb.set_trace()
    with torch.no_grad():
        output, features = resnet(image.cuda())


    pred = output[0].cpu().detach().numpy()
    print(pred)
    # pdb.set_trace()
    # pred_cls = pred.argsort()[-2:][::-1]
    pred_cls= pred.argmax()
    print(i)
    if pred_cls==0:
        class_pred="abnormal"
        # print("abnormal")
    else:
        class_pred = "normal"
        # print("normal")

    feature4 = features[3]


    h_x = F.softmax(output, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    params = list(resnet.parameters())
    weight_softmax = params[-2]

    CAMs = returnCAM(feature4, weight_softmax, image, h_x, pred)
    CAMs = CAMs.detach().cpu().numpy()
    # +++++++++++++++++++++++++++++++++++++
    # pdb.set_trace()

    _, _, h, w = image.size()

    fig = plt.figure(figsize=(8, 12), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    normal_img = np.uint8(255 * inv_normalize(image.squeeze(0)).permute(1, 2, 0).cpu())

    ax1.imshow(normal_img)

    ax2.imshow(np.uint8(255 * inv_normalize(image.squeeze(0)).permute(1, 2, 0).cpu()))

    if pred_cls == 0:  # abnormal
        ax2.text(45, 65, '%s' % class_pred, style='italic',color='white',fontsize=15,
                 bbox={'facecolor': 'red', 'pad': 10})
        ax2.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap='jet', alpha=0.6)
    else: #normal
        ax2.text(45, 65, '%s' % class_pred, style='italic',color='white',fontsize=15,
                 bbox={'facecolor': 'blue', 'pad': 10})

    # normal_img = np.uint8(255 * inv_normalize(image.squeeze(0)).permute(1, 2, 0).cpu())
    #
    # plt.imshow(normal_img)
    #
    # if pred_cls==0: #abnormal
    #     plt.text(45, 65, '%s' % class_pred, style='italic',color='white',fontsize=15,
    #              bbox={'facecolor': 'red', 'pad': 10})
    #     plt.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap='jet', alpha=0.6)
    # else: #normal
    #     plt.text(45, 65, '%s' % class_pred, style='italic',color='white',fontsize=15,
    #              bbox={'facecolor': 'blue', 'pad': 10})


    basename = '%d.jpg'%i
    # basename = basename.split('.')[0] + '_' + object_categories[np.nonzero(label[0])[1,0]] + '.jpg'
    save_image_dir = os.path.join(save_dir,type_dir,basename)
    #
    plt.savefig(save_image_dir)
    plt.close()
    print("saving!")
