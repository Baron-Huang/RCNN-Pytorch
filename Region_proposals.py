import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
from d2l import torch as d2l
from Object_utils import box_center_to_corner, box_corner_to_center, multibox_prior, show_bboxes, box_iou\
    , offset_boxes, assign_anchor_to_bbox, multibox_target, multibox_detection, offset_inverse, nms


if __name__ == '__main__':
    print(np.zeros((3, 2)))
    img = io.imread(r'E:\dog and cat.jpg')

    dog_boxes, cat_boxes = [13.0, 7.0, 190.0, 250], [195.0, 55.0, 321.0, 235.0]
    boxes_test = torch.tensor((dog_boxes, cat_boxes))
    boxes_inter = box_corner_to_center(boxes_test)
    #boxes_2 = box_center_to_corner(boxes_inter)


    h, w = img.shape[:2]
    print(h, w)
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])

    plt.figure(1)
    fig = plt.imshow(img)
    bbox_scale = torch.tensor((w, h, w, h))
    boxes = Y.reshape(h, w, 5, 4)

    show_bboxes(fig.axes, boxes_test, ['dog', 'cat'], 'k')

    show_bboxes(fig.axes, boxes[90, 80, :, :] * bbox_scale,
                ['0', '1'])
    show_bboxes(fig.axes, boxes[145, 245, :, :] * bbox_scale,
                ['2', '3'])
    show_bboxes(fig.axes, boxes[238, 147, 2:3, :] * bbox_scale,
                ['4', 's=0.25, r=1'])
    show_bboxes(fig.axes, boxes[30, 296, 2:3, :] * bbox_scale,
                ['5'])

    boxes_1 = boxes[150, 150, 1, :] * bbox_scale
    boxes_2 = boxes[150, 150, 3, :] * bbox_scale
    boxes_1 = torch.reshape(boxes_1, (1,4))
    boxes_2 = torch.reshape(boxes_2, (1, 4))
    print(box_iou(boxes_1, boxes_2))


    anchor_boxes = torch.cat((boxes[90, 80, :, :], boxes[145, 245, :, :], boxes[238, 147, 2:3, :],
                                boxes[30, 296, 2:3, :]), axis=0)
    boxes_test = boxes_test / bbox_scale
    groundtruth = torch.cat((torch.tensor([0.0, 1.0], dtype=boxes_1.dtype).reshape((2, 1)), boxes_test), axis=1)
    labels = multibox_target(anchor_boxes.unsqueeze(dim=0), groundtruth.unsqueeze(dim=0))
    print(labels[2])
    print(labels[0])
    print(labels[1])

    plt.figure(2)
    fig_2 = plt.imshow(img)
    #show_bboxes(fig_2.axes, boxes_test * bbox_scale, ['dog', 'cat'], 'k')

    show_bboxes(fig_2.axes, anchor_boxes[0:1, :] * bbox_scale,
                ['0-dog'], color_my='black')
    show_bboxes(fig_2.axes, anchor_boxes[1:2, :] * bbox_scale,
                ['4-dog'], color_my='black')
    show_bboxes(fig_2.axes, anchor_boxes[5:7, :] * bbox_scale,
                ['5-cat', '6-cat'], color_my='blue')
    show_bboxes(fig_2.axes, anchor_boxes[9:10, :] * bbox_scale,
                ['9-cat'], color_my='blue')

    plt.figure(3)
    fig_3 = plt.imshow(img)

    anchors = torch.cat((anchor_boxes[0:1, :], anchor_boxes[1:2, :], anchor_boxes[5:7, :],
                         anchor_boxes[9:10, :]), axis=0)
    cls_probs = torch.tensor([[0] * 5,  # 背景的预测概率
                              [0.9, 0.8, 0.3, 0.2, 0.1],  # 狗的预测概率
                              [0.2, 0.1, 0.8, 0.9, 0.85]])  # 猫的预测概率
    show_bboxes(fig_3.axes, anchors * bbox_scale,
                ['dog=0.9', 'dog=0.8', 'cat=0.8', 'cat=0.9', 'cat=0.85'])
    offset_preds = torch.tensor([0] * anchors.numel())

    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                 offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=0.3)

    print(output)

    plt.figure(4)
    fig_4 = plt.imshow(img)
    show_bboxes(fig_4.axes, output[0, 0, 2:].reshape((1, 4)) * bbox_scale,
                ['dog=0.9'], color_my='black')

    show_bboxes(fig_4.axes, output[0, 1, 2:].reshape((1, 4)) * bbox_scale,
                ['cat=0.9'], color_my='blue')

    Y_small = multibox_prior(X, sizes=[0.15], ratios=[1, 2, 0.5])
    Y_small = torch.reshape(Y_small, (1, 271, 354, 3, 4))

    plt.figure(5)
    fig_5 = plt.imshow(img)
    small_region = Y_small[:, 30, 30, :, :]
    for i in range(4):
        for j in range(5):
            small_region = torch.cat((small_region, Y_small[:, 50+61*i, 50+61*j, :, :]), axis=0)

    small_region = small_region[1: , : , :]

    show_bboxes(fig_5.axes, small_region.reshape((20 * 3, 4)) * bbox_scale,
               ['r=1', 'r=2', 'r=0.5'])

    Y_moderate = multibox_prior(X, sizes=[0.3], ratios=[1, 2, 0.5])
    Y_moderate = torch.reshape(Y_moderate, (1, 271, 354, 3, 4))

    plt.figure(6)
    fig_6 = plt.imshow(img)
    moderate_region = Y_moderate[:, 30, 30, :, :]
    for i in range(2):
        for j in range(3):
            moderate_region = torch.cat((moderate_region, Y_moderate[:, 80 + 120 * i, 50 + 120 * j, :, :]), axis=0)

    moderate_region = moderate_region[1:, :, :]

    show_bboxes(fig_6.axes, moderate_region.reshape((6 * 3, 4)) * bbox_scale,
                ['r=1', 'r=2', 'r=0.5'])

    Y_large = multibox_prior(X, sizes=[0.6], ratios=[1, 2, 0.5])
    Y_large= torch.reshape(Y_large, (1, 271, 354, 3, 4))

    plt.figure(7)
    fig_7 = plt.imshow(img)
    large_region = Y_large[:, 30, 30, :, :]
    for i in range(1):
        for j in range(2):
            large_region = torch.cat((large_region, Y_large[:, 120 + 150 * i, 100 + 150 * j, :, :]), axis=0)

    large_region = large_region[1:, :, :]

    show_bboxes(fig_7.axes, large_region.reshape((2 * 3, 4)) * bbox_scale,
                ['r=1', 'r=2', 'r=0.5'])

    plt.show()











