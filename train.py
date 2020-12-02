import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2

import torch.nn.functional as F
# from torchvision.models.resnet import resnet101, resnet50
from PIL import Image
from network.resnet import resnet50

from ai_dataset import AI_Dataset
import pdb
import matplotlib.pyplot as plt


def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)


def one_hot_encoding(val, class_num=20):
    one_hot = []
    for i in range(class_num):
        value = (i == val)
        one_hot.append(value)
    return torch.tensor(one_hot)


def returnCAM(feature_conv, weight_softmax, img,labels,one_hot_labels_erase):
    im_h, im_w = img.size()[2], img.size()[3]
    # print(im_h,im_w)
    # probs, idx = h_x.sort(1, True)


    size_upsample = (im_w, im_h)  #####BE CAREFUL
    batch, nc, h, w = feature_conv.shape
    output_cam = []

    cams = torch.zeros((batch, 21, h,w)).cuda()
    cams.requires_grad = True
    # fgs = torch.zeros((batch,1,im_h,im_w)).cuda()
    # fgs.requires_grad = True

    for i in range(batch):
        for j in range(21):
            # j = j.item()
            # print(erase_idx)

            cam = torch.mm(weight_softmax[j].clone().unsqueeze(0), (feature_conv[i].reshape((nc, h * w))))
            cam = cam.reshape(h, w)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            cams[i, j, :, :] = cam#*labels[i][j] #*h_x[i][j]
    cams = F.upsample(cams,(im_h,im_w),mode='bilinear',align_corners=True)
    torch.cuda.empty_cache()

    pdb.set_trace()
    # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cams


def gen_heatmap(CAMs, batch, height, width):  # CAMs : Batch small_H, small_W
    CAMs_np = CAMs.numpy()
    heatmaps = torch.zeros(batch, height, width, 3)
    for i in range(batch):
        heatmap = cv2.applyColorMap(np.uint8(CAMs_np[i]), cv2.COLORMAP_JET)
        mean_heatmap = np.mean(heatmap)
        # if i==0:
        #
        #     cv2.imwrite('output_heatmap_v%d.jpg' % config.version, heatmap)

        # heatmap = (heatmap[:, :, 2] > int(threshold)) * (heatmap[:, :, 0] < 4)

        # if mean_heatmap > 150:
        #     heatmaps[i, :, :] = torch.from_numpy(heatmap * 0)
        # else:
        heatmaps[i, :, :, :] = torch.from_numpy(heatmap)

    return heatmaps

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_cam(img, cam, path):
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()


def train(config):
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    # transforms.Resize((384, 300)),
    transformations = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.5, 2)),
        # transforms.RandomCrop(384),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.25),
        # transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_img, std=std_img),
    ])

    inv_normalize = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
    )

    dataset = AI_Dataset(mode='train',transform=transformations)



    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,  # config.batch_size
        num_workers=4,#
        shuffle=True,
    )

    test_dataset = AI_Dataset(mode='val', transform=transformations)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,  # config.batch_size
        num_workers=4,  #
        shuffle=False,
    )



    device = "cuda"
    pretrained=False

    resnet = resnet50(pretrained=True).cuda()



    resnet.train()

    # crit_bce = nn.BCEWithLogitsLoss().cuda() #weight=torch.FloatTensor([2]+[1]*3+[1]+[1]*16)
    crit_bce = nn.BCEWithLogitsLoss().cuda()  # weight=torch.FloatTensor([2]+[1]*3+[1]+[1]*16)

    # opt_er = torch.optim.SGD(resnet_er.parameters(), lr=config.lr,momentum=0.9, weight_decay=5e-4,nesterov=False)
    opt= torch.optim.Adam(resnet.parameters(), lr=config.lr, betas=(0.9,0.999), weight_decay=1e-5)

    # min_loss = 999
    best_acc = -999

    torch.autograd.set_detect_anomaly(True)


    print("START TRAINING")
    loss_names = ['c1']
    for epoch in range(config.epoch):
        rl = [0] * 1
        count = 0
        epoch_loss = 0
        right = 0
        wrong = 0
        resnet.train()
        # resnet_cl.train()
        for iteration, (images, labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()
            batch, _, height, width = images.size()
            current_it = epoch * len(train_loader) + iteration
            current_lr = config.lr * pow(1 - (current_it / (len(train_loader) * config.epoch)), 0.9)

            for param_group in opt.param_groups:
                param_group['lr'] = current_lr


            opt.zero_grad()

            # print(labels)
            # print(labels.size())

            outputs_er, features_er = resnet(images)

            # print(outputs_er.size())

            loss = 1 * crit_bce(outputs_er, labels)

            loss.backward()
            opt.step()

            gt = labels[0].cpu().detach().numpy()
            num_cls = len(np.nonzero(gt)[0])
            gt_cls = np.nonzero(gt)[0]
            pred = outputs_er[0].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num_cls:][::-1]


            for c in gt_cls:
                if c in pred_cls:
                    right += 1
                else:
                    wrong += 1

            # print(right)
            # print(wrong)

            # opt_d.step()

            rl[0] += loss.item()

            count += 1

            epoch_loss += loss.item() #+ loss_rec.item()

            torch.cuda.empty_cache()

            if iteration % config.log_step == 0:
                loss_str = ''
                for j in range(len(rl)):
                    loss_str += 'loss_' + loss_names[j]
                    loss_str += ' : ' + "%.4f"%(rl[j] / count)
                    if j != len(rl) - 1:
                        loss_str += ', '
                print("Epoch [%d/%d],Iteration[%s] %s" % (epoch + 1, config.epoch, current_it, loss_str))

                # torch.save(netG.state_dict(),
                #            os.path.join(config.model_path, 'model_netG_v%d.pth' % config.version))
                torch.save(resnet.state_dict(),
                           os.path.join(config.model_path, 'model_resnet_er_v%d.pth' % config.version))
                # torch.save(resnet_cl.state_dict(),
                #            os.path.join(config.model_path, 'model_resnet_cl_v%d.pth' % config.version))

        acc = 100 * (right / (right + wrong))
        print('Train Accuracy : ' + str(round(acc, 4)))

        avg_epoch_loss = epoch_loss / len(train_loader)

        print('Epoch [%d/%d], Loss_c: %.4f,'
              % (epoch + 1, config.epoch, avg_epoch_loss))


        ####Evaluation####
        val_right = 0
        val_wrong = 0

        resnet.eval()
        for val_image, val_label,val_image_name in test_loader:
            with torch.no_grad():

                B, _, _, _ = val_image.size()

                output, _ = resnet(val_image.cuda())

                for i in range(B):
                    gt_cls = []

                    gt = val_label[i].cpu().detach().numpy()
                    num_cls = len(np.nonzero(gt))

                    for j in range(num_cls):
                        gt_cls.append(np.nonzero(gt)[j])

                    pred = output[i].cpu().detach().numpy()
                    pred_cls = pred.argsort()[-num_cls:][::-1]

                    # pdb.set_trace()
                    for c in gt_cls:
                        if c in pred_cls:
                            val_right += 1
                        else:
                            val_wrong += 1
                    # print(val_right)
                    # print(val_wrong)
        val_acc = 100* (val_right/(val_right+val_wrong))

        print("Validation Accuracy:", val_acc)

        if val_acc > best_acc:
            best_acc = val_acc

            # torch.save(netG.state_dict(),
            #            os.path.join(config.model_path, 'model_best_netG_v%d.pth' % config.version))
            torch.save(resnet.state_dict(),
                       os.path.join(config.model_path, 'model_best_resnet_v%d.pth' % config.version))
            # torch.save(netD.state_dict(),
            #            os.path.join(config.model_path, 'model_best_netD_v%d.pth' % config.version))
            print("save best model!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='OWN', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='D:\Github_codes\CAM-master\CAM-master\cifar-10-batches-py')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--epoch', type=int, default=4)

    parser.add_argument('-s', '--save_model_in_epoch', action='store_false')

    parser.add_argument('--resume', type=bool, default=False)

    ########################HYPERPARAMETER##########################
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--weight_c', type=float, default=1)
    parser.add_argument('--weight_c2', type=float, default=1)
    parser.add_argument('--weight_rec', type=float, default=1)
    parser.add_argument('--weight_d', type=float, default=0.5)
    parser.add_argument('--log_step', type=int, default=50)

    config = parser.parse_args()
    print(config)

    train(config)