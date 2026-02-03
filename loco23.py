#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch, cv2, time
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from SSIM import ssim
import faiss
from common23 import get_autoencoder3, get_autoencoder7, get_central_net,  get_student
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--is_train', default=True)
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='./output/ad')
    parser.add_argument('-m', '--model_size', default='medium',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='./pretraining/33/teacher_state.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none', # path to /ImageNet/ILSVRC2012_img_train
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./MVTec-AD',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=50000)
    return parser.parse_args()

# constants
seed = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading 加载数据及变换
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    # seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    if config.is_train:
        print("Running mode: Training", config.model_size)
    else:
        print("Running mode: Testing", config.model_size)

    print("Running epoch:", config.train_steps)

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
        obj_list = [
                    'screw',
                    'capsule',
                    'bottle',
                    'carpet',
                    'leather',
                    'pill',
                    'transistor',
                    'tile',
                    'cable',
                    'zipper',
                    'toothbrush',
                    'metal_nut',
                    'hazelnut',
                    'grid',
                    'wood'
                    ]
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
        obj_list = [
                    'screw_bag',
                    'breakfast_box',
                    'juice_bottle',
                    'pushpins',
                    'splicing_connectors'
                    ]
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = False
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    image_AUROC_list = []
    pixel_AUROC_list = []

    # time_now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    # Data loading
    for config.subdataset in obj_list:
        print("Running dataset:", config.subdataset)
        # create output dir
        train_output_dir = os.path.join(config.output_dir, 'trainings', config.subdataset)
        test_output_dir = os.path.join(config.output_dir, 'anomaly_maps', config.subdataset)
        test_heatmap_dir = os.path.join(config.output_dir, 'heatmap', config.subdataset)
        test_addweight_dir = os.path.join(config.output_dir, 'addweight', config.subdataset)
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
            os.makedirs(test_output_dir)
            os.makedirs(test_heatmap_dir)
            os.makedirs(test_addweight_dir)

        # load data
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'train'),
            transform=transforms.Lambda(train_transform))
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, 'test'))
        gt_set = ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, 'ground_truth'))

        if config.dataset == 'mvtec_ad':
            # mvtec dataset paper recommend 10% validation set
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(seed)
            train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                               [train_size,
                                                                validation_size],
                                                               rng)
        elif config.dataset == 'mvtec_loco':
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, 'validation'),
                transform=transforms.Lambda(train_transform))
        else:
            raise Exception('Unknown config.dataset')

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                                  num_workers=4, pin_memory=True)

        train_loader_infinite = InfiniteDataloader(train_loader)
        validation_loader = DataLoader(validation_set, batch_size=1)

        if pretrain_penalty:
            # load pretraining data for penalty
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * image_size, 2 * image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                      0.225])
            ])
            penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                                   transform=penalty_transform)
            penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                        num_workers=4, pin_memory=True)
            penalty_loader_infinite = InfiniteDataloader(penalty_loader)
        else:
            penalty_loader_infinite = itertools.repeat(None)

        # create models创建模型 包括学生 教师 和两个AE
        teacher = get_central_net(out_channels)
        state_dict = torch.load(config.weights, map_location='cpu')
        teacher.load_state_dict(state_dict)

        student = get_student(out_channels, padding=True)

        autoencoder = get_autoencoder7(out_channels)
        autoencoder2 = get_autoencoder3(out_channels)

        # teacher frozen, others can be trained.
        teacher.eval()
        student.train()
        autoencoder.train()
        autoencoder2.train()

        if on_gpu:
            teacher.cuda()
            student.cuda()
            autoencoder.cuda()
            autoencoder2.cuda()

        # Calculate tensor required for normalization
        teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

        optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                     autoencoder.parameters(),
                                                     autoencoder2.parameters()),
                                     lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)

        if config.is_train:
            tqdm_obj = tqdm(range(config.train_steps))
            for iteration, (image_st, image_ae), image_penalty in zip(
                    tqdm_obj, train_loader_infinite, penalty_loader_infinite):
                if on_gpu:
                    image_st = image_st.cuda()
                    image_ae = image_ae.cuda()
                    if image_penalty is not None:
                        image_penalty = image_penalty.cuda()

                with torch.no_grad():
                    teacher_output_st = teacher(image_st)
                    teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

                student_output_st = student(image_st)[:, :out_channels]
                distance_st = (teacher_output_st - student_output_st) ** 2
                d_hard = torch.quantile(distance_st, q=0.999)
                loss_hard = torch.mean(distance_st[distance_st >= d_hard])

                if image_penalty is not None:
                    student_output_penalty = student(image_penalty)[:, :out_channels]
                    loss_penalty = torch.mean(student_output_penalty ** 2)
                    loss_st = loss_hard + loss_penalty
                else:
                    loss_st = loss_hard

                # ae_output = autoencoder(image_ae)
                ae_output = autoencoder(image_ae)
                ae_output2 = autoencoder2(image_ae)
                with torch.no_grad():
                    teacher_output_ae = teacher(image_ae)
                    teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
                student_output_ae = student(image_ae)[:, out_channels:]
                distance_ae = 0.5 * (teacher_output_ae - ae_output) ** 2 + 0.5 * (teacher_output_ae - ae_output2) ** 2
                distance_stae = 0.5 * (ae_output - student_output_ae) ** 2 + 0.5 * (ae_output2 - student_output_ae) ** 2
                loss_ae = torch.mean(distance_ae)

                d_hard_ae = torch.quantile(distance_stae, q=0.999)
                loss_hard_ae = torch.mean(distance_stae[distance_stae >= d_hard_ae])

                loss_total = loss_st + loss_ae + loss_hard_ae - ssim(teacher_output_ae, ae_output)


                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()

                if iteration % 10 == 0:
                    tqdm_obj.set_description(
                        "Training Current loss: {:.4f}  ".format(loss_total.item()))

                # if iteration % 1000 == 0:
                #     torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
                #     torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
                #     torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

                if iteration % 1000 == 0 and iteration > 0:
                    # run intermediate evaluation
                    teacher.eval()
                    student.eval()
                    autoencoder.eval()
                    autoencoder2.eval()

                    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                        validation_loader=validation_loader, teacher=teacher,
                        student=student, autoencoder=autoencoder, autoencoder2=autoencoder2,
                        teacher_mean=teacher_mean, teacher_std=teacher_std,
                        desc='Intermediate map normalization')
                    auc, auc_pixel = test(
                        gt_set=gt_set, test_set=test_set, teacher=teacher,
                        student=student,
                        autoencoder=autoencoder, autoencoder2=autoencoder2, teacher_mean=teacher_mean,
                        teacher_std=teacher_std,
                        q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                        test_output_dir=test_output_dir, test_heatmap_dir=test_heatmap_dir,
                        test_addweight_dir=test_addweight_dir,
                        desc='Final inference')
                    if iteration == 1000:
                        pixel_AUROC_list.append(auc_pixel)
                        image_AUROC_list.append(auc)
                    else:
                        if auc_pixel > pixel_AUROC_list[-1]:
                            pixel_AUROC_list[-1] = auc_pixel
                        if auc > image_AUROC_list[-1]:
                            image_AUROC_list[-1] = auc
                            torch.save(teacher, os.path.join(train_output_dir, 'teacher_best.pth'))
                            torch.save(student, os.path.join(train_output_dir, 'student_best.pth'))
                            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_best.pth'))
                            torch.save(autoencoder2, os.path.join(train_output_dir, 'autoencoder2_best.pth'))

                    print('Intermediate Image AUROC: {:.4f}'.format(auc))
                    print('best Image AUROC: {:.4f}'.format(image_AUROC_list[-1]))
                    print('Intermediate Pixel AUROC: {:.4f}'.format(auc_pixel))
                    print('best Pixel AUROC: {:.4f}'.format(pixel_AUROC_list[-1]))

                    # teacher frozen
                    teacher.eval()
                    student.train()
                    autoencoder.train()
                    autoencoder2.train()

            teacher.eval()
            student.eval()
            autoencoder.eval()
            autoencoder2.eval()

            torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
            torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))
            torch.save(autoencoder2, os.path.join(train_output_dir, 'autoencoder2_final.pth'))

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder, autoencoder2=autoencoder2,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Final map normalization')

            time0 = time.time()
            auc, auc_pixel = test(
                gt_set=gt_set, test_set=test_set, teacher=teacher,
                student=student,
                autoencoder=autoencoder, autoencoder2=autoencoder2, teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=test_output_dir, test_heatmap_dir=test_heatmap_dir,
                test_addweight_dir=test_addweight_dir,
                desc='Final inference')
            time1 = time.time()

            # pixel_AUROC_list.append(auc_pixel)
            # image_AUROC_list.append(auc)
            if auc_pixel > pixel_AUROC_list[-1]:
                pixel_AUROC_list[-1] = auc_pixel
            if auc > image_AUROC_list[-1]:
                image_AUROC_list[-1] = auc

            print('Inference speed: {:.4f} '.format(len(test_set) / (time1 - time0)))
            print('Final Image AUROC: {:.4f}'.format(auc))
            print('best Image AUROC: {:.4f}'.format(image_AUROC_list[-1]))
            print('Final Pixel AUROC: {:.4f}'.format(auc_pixel))
            print('best Pixel AUROC: {:.4f}'.format(pixel_AUROC_list[-1]))

        else:  # 单独测试部分
            teacher = torch.load(os.path.join(train_output_dir, 'teacher_best.pth'))
            student = torch.load(os.path.join(train_output_dir, 'student_best.pth'))
            autoencoder = torch.load(os.path.join(train_output_dir, 'autoencoder_best.pth'))
            autoencoder2 = torch.load(os.path.join(train_output_dir, 'autoencoder2_best.pth'))
            print("model load successfully")

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder, autoencoder2=autoencoder2,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Final map normalization')

            time0 = time.time()
            auc, auc_pixel = test(
                gt_set=gt_set, test_set=test_set, teacher=teacher,
                student=student,
                autoencoder=autoencoder, autoencoder2=autoencoder2, teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=test_output_dir, test_heatmap_dir=test_heatmap_dir,
                test_addweight_dir=test_addweight_dir,
                desc='Final inference')
            time1 = time.time()

            print('Inference speed: {:.4f} '.format(len(test_set)/(time1 - time0)))
            print('Final Image AUROC: {:.4f}'.format(auc))
            print('Final Pixel AUROC: {:.4f}'.format(auc_pixel))
            print('--------------------------------------------')
            pixel_AUROC_list.append(auc_pixel)
            image_AUROC_list.append(auc)

    print("Image AUROC:")
    print(image_AUROC_list)
    print("Pixel AUROC:")
    print(pixel_AUROC_list)
    print("Image AUROC mean:  " + str(np.mean(image_AUROC_list)))
    print("Pixel AUROC mean:  " + str(np.mean(pixel_AUROC_list)))


# test function and save results
def test(gt_set, test_set, teacher, student, autoencoder, autoencoder2, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end,
         test_output_dir=None, test_heatmap_dir=None, test_addweight_dir=None,
         desc='Running inference'):

    y_true = []
    y_score = []
    mask_cnt = 0
    img_dim = 256
    total_pixel_scores = np.zeros((img_dim * img_dim * len(test_set)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(gt_set)))

    for ((gt_mask, gt_target, gt_path), (image, target, path)) in tqdm(zip(gt_set, test_set)):
        origin_image = image.resize((image_size, image_size))
        origin_image = np.array(origin_image)
        image = default_transform(image)
        image = image[None]

        gt_mask = gt_mask.convert('L')
        gt_mask = gt_mask.resize((image_size, image_size))
        gt_mask = torch.tensor(np.array(gt_mask)).float()
        gt_mask = gt_mask / 255

        image = image.cuda()
        gt_mask = gt_mask.cuda()

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, autoencoder2=autoencoder2, teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        #map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (image_size, image_size), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        heatmap = cv2.convertScaleAbs(map_combined, alpha=255.0)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        add_img = cv2.addWeighted(origin_image, 0.3, heatmap, 0.7, 0)

        map_st = torch.nn.functional.interpolate(
            map_st, (image_size, image_size), mode='bilinear')
        map_st = map_st[0, 0].cpu().numpy()     # 分离 256 256
        heatmap_st = cv2.convertScaleAbs(map_st, alpha=255.0)
        heatmap_st = cv2.applyColorMap(heatmap_st, cv2.COLORMAP_JET)
        add_img_st = cv2.addWeighted(origin_image, 0.3, heatmap_st, 0.7, 0)


        map_ae = torch.nn.functional.interpolate(
            map_ae, (image_size, image_size), mode='bilinear')
        map_ae = map_ae[0, 0].cpu().numpy()     # 分离 256 256
        heatmap_ae = cv2.convertScaleAbs(map_ae, alpha=255.0)
        heatmap_ae = cv2.applyColorMap(heatmap_ae, cv2.COLORMAP_JET)
        add_img_ae = cv2.addWeighted(origin_image, 0.3, heatmap_ae, 0.7, 0)


        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.jpg')
            # file_st = os.path.join(test_output_dir, defect_class, img_nm + '_st.jpg')
            # file_ae = os.path.join(test_output_dir, defect_class, img_nm + '_ae.jpg')
            cv2.imwrite(file, map_combined*255)
            # cv2.imwrite(file_st, map_st * 255)
            # cv2.imwrite(file_ae, map_ae * 255)
        if test_heatmap_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_heatmap_dir, defect_class)):
                os.makedirs(os.path.join(test_heatmap_dir, defect_class))
            file1 = os.path.join(test_heatmap_dir, defect_class, img_nm + '.jpg')
            # file1_st = os.path.join(test_heatmap_dir, defect_class, img_nm + '_st.jpg')
            # file1_ae = os.path.join(test_heatmap_dir, defect_class, img_nm + '_ae.jpg')
            cv2.imwrite(file1, heatmap)
            # cv2.imwrite(file1_st, heatmap_st)
            # cv2.imwrite(file1_ae, heatmap_ae)
        if test_addweight_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_addweight_dir, defect_class)):
                os.makedirs(os.path.join(test_addweight_dir, defect_class))
            file2 = os.path.join(test_addweight_dir, defect_class, img_nm + '.jpg')
            # file2_st = os.path.join(test_addweight_dir, defect_class, img_nm + '_st.jpg')
            # file2_ae = os.path.join(test_addweight_dir, defect_class, img_nm + '_ae.jpg')
            cv2.imwrite(file2, add_img)
            # cv2.imwrite(file2_st, add_img_st)
            # cv2.imwrite(file2_ae, add_img_ae)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        gt_mask_cv = gt_mask.cpu().numpy()
        flat_gt_mask = gt_mask_cv.flatten()
        flat_out_mask = map_combined.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_gt_mask
        mask_cnt += 1

    auc = roc_auc_score(y_true=y_true, y_score=y_score)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    return auc * 100, auroc_pixel * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, autoencoder2 ,teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)

    autoencoder_output = autoencoder(image)
    autoencoder_output2 = autoencoder2(image)

    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((0.5 * (autoencoder_output - student_output[:, out_channels:]) ** 2 + 0.5 * (autoencoder_output2 -student_output[:, out_channels:]) ** 2),
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder, autoencoder2,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, autoencoder2=autoencoder2, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def off_diagonal(x):
    n, m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n -1, n + 1)[:, 1:].flatten()

if __name__ == '__main__':
    main()