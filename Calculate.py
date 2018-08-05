# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2


def match(resultsfile, groundtruthfile):
    results, num_detectedbox = load(resultsfile)
    groundtruth, num_groundtruthbox = load(groundtruthfile)

    assert len(results) == len(groundtruth), "数量不匹配: groundtruth中图片数量为%d，而检测结果中图片数量为%d" % (
    len(groundtruth), len(results))
    maxiou_confidence = np.array([])
    for i in range(len(results)):
        print(results[i][0])
        # fname = './' + results[i][0] + '.jpg'
        # image = cv2.imread(fname)
        for j in range(2, len(results[i])):
            iou_array = np.array([])
            detectedbox = results[i][j]
            confidence = detectedbox[-1]
            # x_min, y_min = int(detectedbox[0]), int(detectedbox[1])
            # x_max = int(detectedbox[0] + detectedbox[2])
            # y_max = int(detectedbox[1] + detectedbox[3])
            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            for k in range(2, len(groundtruth[i])):
                groundtruthbox = groundtruth[i][k]
                iou = cal_IoU(detectedbox, groundtruthbox)
                iou_array = np.append(iou_array, iou)
                # x_min, y_min = int(groundtruthbox[0]), int(groundtruthbox[1])
                # x_max = int(groundtruthbox[0] + groundtruthbox[2])
                # y_max = int(groundtruthbox[1] + groundtruthbox[3])
                # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            maxiou = np.max(iou_array)
            maxiou_confidence = np.append(maxiou_confidence, [maxiou, confidence])
        # cv2.imshow("Image",image)
        # cv2.waitKey()
    maxiou_confidence = maxiou_confidence.reshape(-1, 2)
    maxiou_confidence = maxiou_confidence[np.argsort(-maxiou_confidence[:, 1])]

    return maxiou_confidence, num_detectedbox, num_groundtruthbox


def thres(maxiou_confidence, threshold = 0.5):
    maxious = maxiou_confidence[:, 0]
    confidences = maxiou_confidence[:, 1]
    true_or_flase = (maxious > threshold)
    tf_confidence = np.array([true_or_flase, confidences])
    tf_confidence = tf_confidence.T
    tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
    return tf_confidence


def plot(x_confidence, num_groundtruthbox):
    fp_list = []
    recall_list = []
    precision_list = []
    for num in range(len(x_confidence)):
        arr = x_confidence[:(num + 1), 0]# 注意要加1
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        recall = tp / num_groundtruthbox
        precision = tp / (tp + fp)

        fp_list.append(fp)
        recall_list.append(recall)
        precision_list.append(precision)

    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    plt.plot(fp_list, recall_list)

    plt.figure()
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.plot(recall_list, precision_list)
    plt.show()


def ellipse_to_rect(ellipse):
    """
    将椭圆框转换为水平竖直的矩形框
    :param ellipse: [major_axis_radius minor_axis_radius angle center_x center_y, score]
    :return: [leftx, topy, width, height, score]
    """
    major_axis_radius, minor_axis_radius, angle, center_x, center_y, score = ellipse
    leftx = center_x - minor_axis_radius
    topy = center_y - major_axis_radius
    width = 2 * minor_axis_radius
    height = 2 * major_axis_radius
    rect = [leftx, topy, width, height, score]
    return rect


def cal_IoU(detectedbox, groundtruthbox):
    """
    计算两个水平竖直的矩形的交并比
    :param detectedbox: [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return: iou
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    #plt.figure()

    #print('det',detectedbox)
    #print('gt',groundtruthbox)

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iou = intersection / union
        print(iou)
        return iou
    else:
        return 0


def load(txtfile):
    '''
    读取检测结果或 groundtruth 的文档，若为椭圆坐标，转换为矩形坐标
    :param txtfile: 读入的 .txt 文件，格式要求与 FDDB 相同
    :return: imagelist列表, 每张图片的信息在一行, 第一列是图片名称, 第二列是人脸个数,
    后面的列均为列表, 包含 4 个矩形坐标和 1 个score
    num_allboxes: 矩形框的总个数
    '''
    imagelist = []
    txtfile = open(txtfile, 'r')
    lines = txtfile.readlines()
    num_allboxes = 0
    i = 0
    while i < len(lines):
        image = []
        image.append(lines[i].strip())
        num_faces = int(lines[i + 1])
        num_allboxes = num_allboxes + num_faces
        image.append(num_faces)
        if num_faces > 0:
            for num in range(num_faces):
                boundingbox = lines[i + 2 + num].strip()
                boundingbox = boundingbox.split()
                boundingbox = list(map(float, boundingbox))
                if len(boundingbox) == 6:
                    boundingbox = ellipse_to_rect(boundingbox)
                image.append(boundingbox)
        imagelist.append(image)
        i = i + num_faces + 2
    txtfile.close()
    return imagelist, num_allboxes

