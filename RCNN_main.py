import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from skimage import io, transform
import torch
from Object_utils import box_center_to_corner, box_corner_to_center, multibox_prior, show_bboxes, box_iou\
    , offset_boxes, assign_anchor_to_bbox, multibox_target, multibox_detection, offset_inverse, nms,\
    read_objective_detection_datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import xlrd
from RCNN_fit_function import cls_model_fit, acc_scores, regression_model_fit
from RCNN_model import VGG_Net, Regression_Net, VGG_Extractor, VGG_Probability
from torchvision.models import vgg11
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    #val_loader, _, _ = read_objective_detection_datasets(
    #                img_dir=r'F:\RCNN\Datasets\banana-detection\bananas_val\images',
    #                label_dir=r'F:\RCNN\Datasets\banana-detection\bananas_val\label.xls',
    #                img_size=[224, 224], batch_size=16, img_num=100)

    ###Show Results
    '''
    img_num = 2
    for val_img, val_label in val_loader:
        pass
    img = val_img[img_num]
    img = np.transpose(img, (1, 2, 0))
    img = img.numpy()
    plt.figure(1)
    fig_1 = plt.imshow(img)
    show_bboxes(fig_1.axes, val_label[img_num, 1:].unsqueeze(dim=0), color_my='white')

    anchor_boxes = multibox_prior(val_img[img_num], sizes=[0.2], ratios=[1, 2, 0.5])

    anchor_boxes = torch.reshape(anchor_boxes, (1, 224, 224, 3, 4))
    moderate_anchor = anchor_boxes[:, 30, 30, :, :]
    for i in range(7):
        for j in range(7):
            moderate_anchor = torch.cat(
                (moderate_anchor, anchor_boxes[:, 25 + 30 * i,  25 + 30 * j, :, :]),
                axis=0)

    bbox_scale = [224.0, 224.0, 224.0, 224.0]
    bbox_scale = torch.tensor(bbox_scale)
    moderate_anchor = moderate_anchor[1:, :, :]
    moderate_anchor = moderate_anchor.reshape((49 * 3, 4)) * bbox_scale
    moderate_anchor[moderate_anchor < 0] = 0
    moderate_anchor[moderate_anchor > 223] = 223

    plt.figure(2)
    fig_2 = plt.imshow(img)
    show_bboxes(fig_2.axes, moderate_anchor, ['r=1', 'r=2', 'r=0.5'])

    plt.figure(3)
    plt.imshow(img[int(moderate_anchor[2][0]):int(moderate_anchor[2][2]),
               int(moderate_anchor[2][1]):int(moderate_anchor[2][3]), :])
    groundtruth = val_label[img_num, :]
    groundtruth[1:] = groundtruth[1:] / 224.0

    #labels = multibox_target(moderate_anchor.unsqueeze(dim=0) / bbox_scale,
     #                        groundtruth.unsqueeze(dim=0).unsqueeze(dim=0), iou_threshold=0.2)
    #assign_cls = labels[2].numpy()
    #target_index = torch.nonzero(labels[2])[:, 1]

    iou_values = []
    target_values = []
    offset_values = []
    for j in range(moderate_anchor.shape[0]):
        iou_values.append(box_iou(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                                  groundtruth[1:].unsqueeze(dim=0)))
        if (box_iou(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                                  groundtruth[1:].unsqueeze(dim=0))) >= 0.05:
            target_values.append(j)
            offset_values.append(offset_boxes(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                                  groundtruth[1:].unsqueeze(dim=0)))
    assign_cls = torch.zeros((moderate_anchor.shape[0]))
    assign_cls[target_values] = 1


    plt.figure(4)
    fig_4 = plt.imshow(img)
    show_bboxes(fig_4.axes, moderate_anchor[target_values,:])
    plt.show()

    for i in range(moderate_anchor.shape[0]):
        io.imsave(
        'F:\RCNN\Datasets\Region_proposals\Region_train'+'\\'+str(int(assign_cls[i]))+'\\'
            +str(i)+'.jpg', img[int(moderate_anchor[i][1]):int(moderate_anchor[i][3]),
            int(moderate_anchor[i][0]):int(moderate_anchor[i][2]), :]
        )

    np.save('F:\RCNN\Datasets\Region_proposals\Region_train'+'\\'+'offset.npy',
            np.array(offset_values))
    '''

    ###1-th step:Processing Region proposal image
    '''
    val_img = torch.zeros((1, 3, 224, 224))
    val_label = torch.zeros((1, 5))
    for img, label in val_loader:
        val_img = torch.cat((val_img, img))
        val_label = torch.cat((val_label, label))

    val_img = val_img[1:, :, :, :]
    val_label = val_label[1:, :]
    val_label = val_label.numpy() * 256.0 / 224.0
    img = val_img[2]
    img = np.transpose(img, (1, 2, 0))
    img = img.numpy()

    plt.figure(1)
    fig_1 = plt.imshow(img)
    plt.show()
    '''

    '''
    count = 0
    offset_sum = torch.zeros((1, 4))
    anchor_sum = torch.zeros((1, 4))
    cls_sum = torch.zeros((1, ))
    groundtruth_boxes = torch.zeros((1, 4))
    for val_img, val_label in val_loader:
        for i in range(val_img.shape[0]):
            img = val_img[i]
            img = np.transpose(img, (1, 2, 0))
            img = img.numpy()

            print(i)

            #plt.figure(1)
            #fig_1 = plt.imshow(img)
            #show_bboxes(fig_1.axes, val_label[i, 1:].unsqueeze(dim=0), color_my='white')
            #plt.show()

            anchor_boxes = multibox_prior(val_img[i], sizes=[0.2], ratios=[1, 2, 0.5])

            anchor_boxes = torch.reshape(anchor_boxes, (1, 224, 224, 3, 4))
            moderate_anchor = anchor_boxes[:, 30, 30, :, :]
            for s in range(7):
                for w in range(7):
                    moderate_anchor = torch.cat(
                        (moderate_anchor, anchor_boxes[:, 25 + 30 * s, 25 + 30 * w, :, :]), axis=0)

            bbox_scale = [224.0, 224.0, 224.0, 224.0]
            bbox_scale = torch.tensor(bbox_scale)
            moderate_anchor = moderate_anchor[1:, :, :]
            moderate_anchor = moderate_anchor.reshape((49 * 3, 4)) * bbox_scale
            moderate_anchor[moderate_anchor < 0] = 0
            moderate_anchor[moderate_anchor > 223] = 223

            groundtruth = val_label[i, :]
            groundtruth[1:] = groundtruth[1:] / 224.0

            iou_values = []
            target_values = []
            offset_values = torch.zeros((1, 4))
            for j in range(moderate_anchor.shape[0]):
                iou_values.append(box_iou(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                                          groundtruth[1:].unsqueeze(dim=0)))
                offset_values = torch.cat((offset_values,
                                          offset_boxes(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                                            groundtruth[1:].unsqueeze(dim=0))), axis=0)
                if (box_iou(moderate_anchor[j, :].unsqueeze(dim=0) / bbox_scale,
                            groundtruth[1:].unsqueeze(dim=0))) >= 0.1:
                    target_values.append(j)

            assign_cls = torch.zeros((moderate_anchor.shape[0]))
            assign_cls[target_values] = 1
            offset_values = offset_values[1:, :]

            del_count = 0
            extract_num = []
            for j in range(moderate_anchor.shape[0]):
                if int(assign_cls[j]) == 1:
                    extract_num.append(j)
                else:
                    del_count += 1
                    if del_count % 5 == 0:
                        extract_num.append(j)
                    else:
                        pass

            moderate_anchor = moderate_anchor[extract_num, :]
            assign_cls = assign_cls[extract_num]
            offset_values = offset_values[extract_num]

            offset_sum = torch.cat((offset_sum, offset_values))
            cls_sum = torch.cat((cls_sum, assign_cls))
            anchor_sum = torch.cat((anchor_sum, moderate_anchor))
            for j in range(moderate_anchor.shape[0]):
                groundtruth_boxes = torch.cat((groundtruth_boxes, val_label[i, 1:].unsqueeze(dim=0)))

            if count < 900:
                for k in range(moderate_anchor.shape[0]):
                    io.imsave(
                        'F:\Object_Detection\Datasets\Cls_model_data\Test' + '\\' +
                        str(int(assign_cls[k])) + '\\' + str(count) + '-' + str(k) + '.jpg',
                        img[int(moderate_anchor[k][1]):int(moderate_anchor[k][3]),
                        int(moderate_anchor[k][0]):int(moderate_anchor[k][2]), :]
                    )

                    io.imsave(
                        'F:\Object_Detection\Datasets\Test\images'
                        + '\\' + str(count) + '-' + str(k) + '.jpg',
                        img[int(moderate_anchor[k][1]):int(moderate_anchor[k][3]),
                        int(moderate_anchor[k][0]):int(moderate_anchor[k][2]), :]
                    )
            else:
                for k in range(moderate_anchor.shape[0]):
                    io.imsave(
                        'F:\Object_Detection\Datasets\Cls_model_data\Test' + '\\' +
                        str(int(assign_cls[k])) + '\\' + str(count) + '-' + str(k) + '.jpg',
                        img[int(moderate_anchor[k][1]):int(moderate_anchor[k][3]),
                        int(moderate_anchor[k][0]):int(moderate_anchor[k][2]), :]
                    )

                    io.imsave(
                        'F:\Object_Detection\Datasets\Test\images'
                        + '\\' + str(count) + '-' + str(k) + '.jpg',
                        img[int(moderate_anchor[k][1]):int(moderate_anchor[k][3]),
                        int(moderate_anchor[k][0]):int(moderate_anchor[k][2]), :]
                    )

            count += 1
            print(count)

    offset_sum = offset_sum[1:, :]
    cls_sum = cls_sum[1:]
    anchor_sum = anchor_sum[1:, :]
    groundtruth_boxes = groundtruth_boxes[1:, :]
    np.save('F:\Object_Detection\Datasets\Test\\offset_sum.npy', offset_sum.numpy())
    np.save('F:\Object_Detection\Datasets\Test\\cls_sum.npy', cls_sum.numpy())
    np.save('F:\Object_Detection\Datasets\Test\\anchor_sum.npy', anchor_sum.numpy())
    np.save('F:\Object_Detection\Datasets\Test\\groundtruth_boxes.npy', groundtruth_boxes.numpy())
    '''

    ###2-th step: Training classification model
    '''
    setup_seed(1234)

    transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()
                                       , transforms.Normalize(mean=0.5, std=0.5)])

    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
    train_data = ImageFolder(r'F:\RCNN\Datasets\Cls_model_data\Train',
                             transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=1)

    val_data = ImageFolder(r'F:\RCNN\Datasets\Cls_model_data\Val',
                             transform=transform)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=1)

    test_data = ImageFolder(r'F:\RCNN\Datasets\Cls_model_data\Test',
                            transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=1)


    base_net = vgg11(pretrained=False)
    vgg_net = VGG_Net(class_num=2, base_net=base_net)
    vgg_net = vgg_net.cuda(0)
    cls_model_fit(ddai_net=vgg_net, train_loader=train_loader, val_loader=val_loader,
                  test_loader=test_loader, lr_fn='normal', epoch=50)
    model_weights = torch.load(r'F:\RCNN\Weights\Vgg_net.pt',map_location='cuda:0')
    vgg_net.load_state_dict(model_weights, strict=True)

    acc_scores(model=vgg_net, data_loader=test_loader, gpu_device=0, out_mode='single', class_num = 2)
    '''

    ###3-th step: Extracting feature vector and probability vector of Anchor image
    ##3.1 Extracting feature vector
    '''
    base_net = vgg11(pretrained=False)
    vgg_extractor = VGG_Extractor(base_net = base_net)
    model_weights = torch.load(r'F:\RCNN\Weights\Vgg_net.pt', map_location='cuda:0')
    vgg_extractor.load_state_dict(model_weights, strict=False)
    vgg_extractor = vgg_extractor.cuda(0)

    transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()
                                       , transforms.Normalize(mean=0.5, std=0.5)])

    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
    test_data = ImageFolder(r'F:\RCNN\Datasets\Val',
                             transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=1)

    feature_vector = torch.zeros((1, 512)).cuda(0)
    with torch.no_grad():
        for anchor_img , _ in test_loader:
            anchor_img = anchor_img.cuda(0)
            feature_vector = torch.cat((feature_vector, vgg_extractor(anchor_img)))

    feature_vector = feature_vector[1:, :]
    print(feature_vector.shape)

    np.save('F:\RCNN\Datasets\Val\\val_feature.npy',
            feature_vector.cpu().detach().numpy())
    '''


    ##3.2 Extracting probability vector of Anchor image
    '''
    path = r'F:\RCNN\Datasets\Val\images'
    file_list = os.listdir(path)

    for file in file_list:
        # 补0 4表示补0后名字共4位 针对imagnet-1000足以
        for i in range(len(file)):
            if file[i] == '-':
                filename_1 = file[:i+1]
                filename_2 = file[i+1:]
                filename_1 = filename_1.zfill(4)
                filename_2 = filename_2.zfill(6)
                filename = filename_1 + filename_2
        #print(filename)
        new_name = ''.join(filename)
        os.rename(path + '\\' + file, path + '\\' + new_name)
    '''

    '''
    base_net = vgg11(pretrained=False)
    vgg_probability = VGG_Probability(base_net=base_net, class_num=2)
    model_weights = torch.load(r'F:\RCNN\Weights\Vgg_net.pt', map_location='cuda:0')
    vgg_probability.load_state_dict(model_weights, strict=True)
    vgg_probability = vgg_probability.cuda(0)


    transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()
                                       , transforms.Normalize(mean=0.5, std=0.5)])

    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
    test_data = ImageFolder(r'F:\RCNN\Datasets\Test',
                            transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=1)

    probability_vector = torch.zeros((1, 2)).cuda(0)
    with torch.no_grad():
        for anchor_img, _ in test_loader:
            anchor_img = anchor_img.cuda(0)
            probability_vector = torch.cat((probability_vector, vgg_probability(anchor_img)))

    probability_vector = probability_vector[1:, :]
    print(probability_vector.shape)
    probability_vector = probability_vector.cpu().detach().numpy()
    pre_label = np.argmax(probability_vector, axis = 1)
    ture_label = np.load(r'F:\Object_Detection\Datasets\Test\cls_sum.npy', allow_pickle=True)

    score = 0
    for i in range(pre_label.shape[0]):
        if pre_label[i] == ture_label[i]:
            score += 1
    print(score / 4723)

    np.save('F:\RCNN\Datasets\Test\\test_probability.npy', probability_vector)
    '''


    ###4-th step: Training regression model
    ##4.1 processing Train-set metric
    '''
    offset_sum = np.load(r'F:\Object_Detection\Datasets\Train-Val-metirc\offset_sum.npy',
                            allow_pickle=True)
    cls_sum = np.load(r'F:\Object_Detection\Datasets\Train-Val-metirc\cls_sum.npy',
                            allow_pickle=True)
    groundtruth_boxes = np.load(r'F:\Object_Detection\Datasets\Train-Val-metirc\groundtruth_boxes.npy',
                            allow_pickle=True)
    anchor_sum = np.load(r'F:\Object_Detection\Datasets\Train-Val-metirc\anchor_sum.npy',
                            allow_pickle=True)

    train_anchor = anchor_sum[:39093, :]
    val_anchor = anchor_sum[39093:, :]
    np.save('F:\Object_Detection\Datasets\Train\\train_anchor_sum.npy', train_anchor)
    np.save('F:\Object_Detection\Datasets\\Val\\val_anchor_sum.npy', val_anchor)

    train_groundtruth_boxes = groundtruth_boxes[:39093, :]
    val_groundtruth_boxes = groundtruth_boxes[39093:, :]
    np.save('F:\Object_Detection\Datasets\Train\\train_groundtruth_boxes_sum.npy', train_groundtruth_boxes)
    np.save('F:\Object_Detection\Datasets\\Val\\val_groundtruth_boxes_sum.npy', val_groundtruth_boxes)

    train_cls_sum = cls_sum[:39093]
    val_cls_sum = cls_sum[39093:]
    np.save('F:\Object_Detection\Datasets\Train\\train_anchor_cls_sum.npy', train_cls_sum)
    np.save('F:\Object_Detection\Datasets\\Val\\val_anchor_cls_sum.npy', val_cls_sum)

    train_offset_sum = offset_sum[:39093, :]
    val_offset_sum = offset_sum[39093:, :]
    np.save('F:\Object_Detection\Datasets\Train\\train_offset_sum.npy', train_offset_sum)
    np.save('F:\Object_Detection\Datasets\\Val\\val_offset_sum.npy', val_offset_sum)
    '''

    ##4.2 show the detail of those metrics
    '''
    cls = np.load(r'F:\Object_Detection\Datasets\Val\val_anchor_sum.npy', allow_pickle=True)
    pro = np.load(r'F:\Object_Detection\Datasets\Val\val_groundtruth_boxes_sum.npy', allow_pickle=True) * 224
    pro = torch.tensor(pro)
    cls = torch.tensor(cls)
    plt.figure(1)
    img = io.imread(r'F:\Object_Detection\Datasets\banana-detection\bananas_train\images\900.png')
    img = transform.resize(img, (224, 224))
    flg = plt.imshow(img)
    show_bboxes(flg.axes, pro[0, :].unsqueeze(dim=0))


    offset = pro / 224.0 - cls / 224.0
    np.save(r'F:\Object_Detection\Datasets\Val\val_offset_sum.npy', offset.numpy())
    plt.show()
    print(torch.max(offset))
    '''

    ##4.3 fitting regression model
    '''
    reg_net = Regression_Net(regression_num=4)
    reg_net = reg_net.cuda(0)

    features = np.load(r'F:\Object_Detection\Datasets\Train\train_feature.npy', allow_pickle=True)
    anchor = np.load(r'F:\Object_Detection\Datasets\Train\anchor_sum.npy', allow_pickle=True)
    anchor = anchor / 224.0
    true_boxes = np.load(r'F:\Object_Detection\Datasets\Train\groundtruth_boxes.npy', allow_pickle=True)
    offset = true_boxes - anchor
    train_offset = offset[:42740, :]
    train_offset = torch.tensor(train_offset)
    features = torch.tensor(features)

    val_features = np.load(r'F:\Object_Detection\Datasets\Val\val_feature.npy', allow_pickle=True)
    val_offset = offset[42740:, :]
    val_offset = torch.tensor(val_offset)
    val_features = torch.tensor(val_features)

    reg_data = TensorDataset(features, train_offset)
    val_data = TensorDataset(val_features, val_offset)
    reg_loader = DataLoader(reg_data, batch_size=256, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=1)
    #regression_model_fit(ddai_net=reg_net, train_loader=reg_loader, val_loader=val_loader,
    #                     test_loader=val_loader,
    #                     lr_fn='normal', epoch = 50, gpu_device = 0)

    model_weights = torch.load(r'F:\Object_Detection\Weights\Regression_net.pt', map_location='cuda:0')
    reg_net.load_state_dict(model_weights, strict=True)
    with torch.no_grad():
        pre_val_offset = reg_net(val_features.cuda(0))
        pre_train_offset = reg_net(features.cuda(0))
    pre_val_offset = pre_val_offset.cpu().detach().numpy()
    pre_train_offset = pre_train_offset.cpu().detach().numpy()
    pre_boxes = anchor[42740:, :] + pre_val_offset

    np.save(r'F:\Object_Detection\Datasets\Train\pre_train_offset.npy', pre_train_offset)
    print(np.mean(np.abs(pre_boxes - true_boxes[42740:, :])))

    print()
    '''

    ##4.4 testing(show results)

    train_anchor = np.load(r'F:\RCNN\Datasets\Train\anchor_sum.npy', allow_pickle=True)
    train_groudtruth = np.load(r'F:\RCNN\Datasets\Train\groundtruth_boxes.npy',
                               allow_pickle=True)

    plt.figure(1)
    img = io.imread(r'F:\RCNN\Datasets\banana-detection\bananas_train\images\400.png')
    img = skimage.transform.resize(img, (224, 224))
    fig = plt.imshow(img)

    search_num = np.arange(18943, 18943+47)
    true_num = 18943+20
    show_bboxes(fig.axes, torch.tensor(train_groudtruth[true_num, :] * 224.0).unsqueeze(dim=0))


    plt.figure(2)
    fig_2 = plt.imshow(img)
    show_bboxes(fig_2.axes, torch.tensor(train_anchor[search_num, :]), color_my='white')
    pre_cls =  np.load(r'F:\RCNN\Datasets\Train\train_probability.npy', allow_pickle=True)
    true_cls = np.load(r'F:\RCNN\Datasets\Train\cls_sum.npy', allow_pickle=True)
    pre_train_offset = np.load(r'F:\RCNN\Datasets\Train\pre_train_offset.npy', allow_pickle=True)
    ins_cls = pre_cls[search_num,:]
    ins_true = true_cls[search_num]
    pre_offset = pre_train_offset[search_num, :]

    num = np.argmax(ins_cls, axis=1)
    ins_num = np.nonzero(num)


    show_a = train_anchor[search_num, :]
    final_show = show_a[ins_num, :]
    xx = train_anchor[search_num, :]
    xxx = xx[ins_num, :]
    new = multibox_detection(torch.tensor(final_show),
                             torch.tensor([0, 0, 0, 0]).unsqueeze(dim=0),
                             torch.tensor(xxx / 224.0),
                             nms_threshold=0.01)

    final_show_non = np.mean(final_show, axis=1)
    plt.figure(3)
    fig_3 = plt.imshow(img)
    show_bboxes(fig_3.axes, torch.tensor(final_show_non[:, :]))

    plt.figure(4)
    fig_4 = plt.imshow(img)
    pre_true_off = pre_offset[ins_num, :]
    check_loc = final_show + pre_true_off
    check_final = np.mean(check_loc, axis=1)
    show_bboxes(fig_4.axes, torch.tensor(check_final))

    plt.figure(5)
    fig_5 = plt.imshow(img)
    show_bboxes(fig_5.axes, new[0, :, 2:] * 224.0)

    plt.show()








