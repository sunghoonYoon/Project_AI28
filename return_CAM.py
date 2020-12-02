
import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from network.resnet import resnet50
from ai_dataset import AI_Dataset

import glob
import tqdm
from PIL import Image
import pdb
import random
import shutil
import matplotlib.pyplot as plt



def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    # test_loader, num_class = utils.get_testloader(config.dataset,
    #                                     config.dataset_path,
    #                                     config.img_size)

    # hook


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
            cams[i, j, :, :] = cam*labels[i][j]
            cam_dict[j] =  cv2.resize(cam.cpu().detach().numpy(),size_upsample)

    cams = F.upsample(cams,(im_h,im_w),mode='bilinear',align_corners=False)


    # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cams

if __name__=='__main__':

    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    object_categories= ['abnormal','normal']

    transformations = transforms.Compose([
        # transforms.RandomResizedCrop(384,scale=(0.8,1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_img, std=std_img),
    ])


    inv_normalize = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
    )

    dataset = AI_Dataset(mode='val',transform=transformations)

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  # config.batch_size
        num_workers=4,  #
        shuffle=True,
    )

    device = "cuda"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    resnet =resnet50(pretrained=False).cuda()
    resnet.load_state_dict(torch.load("C:\\Users\\sunghoon Yoon\\PycharmProjects\\AI28\\model\\model_resnet_er_v3.pth"))

    # resnet = torch.nn.DataParallel(resnet50_ysh2(pretrained=False,norm_layer=SynchronizedBatchNorm2d).cuda())
    # resnet.load_state_dict(torch.load("/mnt/usb0/shyoon/CAM/model/model_resnet_er_v%d_sgd.pth" % version))
    # resnet.load_state_dict(torch.load("/mnt/usb0/shyoon/CAM/model/model_best_resnet_v%d.pth" % version))
    resnet.eval()




    # img_dir = '/mnt/usb0/shyoon/CAM/data/VOCdevkit/VOC2012/JPEGImages/2008_*.jpg'
    # images = glob.glob(img_dir)
    root_dir = 'C:\\Users\\sunghoon Yoon\\PycharmProjects\\AI28\\output'

    save_root_dir = os.path.join(root_dir,'segment')



    if os.path.isdir(save_root_dir):
        shutil.rmtree(save_root_dir + "/", ignore_errors=True)
    else:
        os.mkdir(save_root_dir)
        print("make directory")

    for iteration,(image,label,image_dir) in enumerate(val_loader):

        # pdb.set_trace()
        batch,_,height,width = image.size()

        image= image.cuda()
        label = label.cuda()

        print(image_dir)
        print(label)

        image_dir=image_dir[0]

        basename = os.path.basename(image_dir)

 # 375 500

        with torch.no_grad():
            resnet.eval()
            outputs_v, features = resnet(image)

            feature4 = features[3]
        # pdb.set_trace()
        #     feature4 = F.interpolate(feature4,(height,width),mode='bilinear',align_corners=True)

            # feature4 = feature4.squeeze(0).permute(1,2,0).detach().cpu().numpy()


            h_x = F.softmax(outputs_v, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)

            print(outputs_v)

            params = list(resnet.parameters())
            weight_softmax = params[-2]

            # max_norm()

            # CAMs = F.relu(F.interpolate(feature4.unsqueeze(0),(height,width),mode='bilinear',align_corners=True))
            # # CAMs = F.sigmoid(F.interpolate(feature4, (height, width), mode='bilinear', align_corners=True))
            #
            # # CAMs = CAMs-torch.min(CAMs)
            # CAMs = CAMs/torch.max(CAMs)
            #
            # # CAMs = CAMs1+CAMs2
            # # CAMs = CAMs/ (torch.max(CAMs)/2)
            # # pdb.set_trace()
            # # CAMs[CAMs>=1]=1
            # # CAMs[CAMs<=0]=0
            # # CAMs = F.sigmoid(CAMs)
            #
            # CAMs = CAMs.detach().cpu().numpy()
            CAMs = returnCAM(feature4, weight_softmax, image, h_x,label)
        #+++++++++++++++++++++++++++++++++++++
        # pdb.set_trace()

        _,_,h,w = image.size()

        fig = plt.figure(figsize=(8,12),dpi=1000)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # plt.subplot(1,2,1)

        # print()
        normal_img = np.uint8(255*inv_normalize(image.squeeze(0)).permute(1,2,0).cpu())

        ax1.imshow(normal_img)

        ax2.imshow(np.uint8(255*inv_normalize(image.squeeze(0)).permute(1,2,0).cpu()))
        ax2.imshow(np.uint8(255*CAMs[0][idx[0]]), cmap='jet',alpha=0.6)
        # plt.imshow(np.uint8(255 * (CAMs[0][np.nonzero(label[0])[1,0]])), cmap='jet', alpha=0.6)
        # plt.imshow(np.uint8(255 * feature4[:,:,np.nonzero(labels[0])[1, 0]]), cmap='jet', alpha=0.6)
        #
        basename = basename.split('.')[0]+'_'+object_categories[idx[0]]+'.jpg'
        # basename = basename.split('.')[0] + '_' + object_categories[np.nonzero(label[0])[1,0]] + '.jpg'
        save_dir = os.path.join(save_root_dir, basename)
        #
        plt.savefig(save_dir)
        print("saving!")
