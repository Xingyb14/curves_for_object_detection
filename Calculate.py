# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2


def match(resultsfile, groundtruthfile, show_images):
    """
    匹配检测框和标注框, 为每一个检测框得到一个最大交并比   
    :param resultsfile: 包含检测结果的.txt文件
    :param groundtruthfile: 包含标准答案的.txt文件
    :param show_images: 是否显示图片
    :return maxiou_confidence: np.array, 存放所有检测框对应的最大交并比和置信度
    :return num_detectedbox: int, 检测框的总数
    :return num_groundtruthbox: int, 标注框的总数
    """
    results, num_detectedbox = load(resultsfile)
    groundtruth, num_groundtruthbox = load(groundtruthfile)

    assert len(results) == len(groundtruth), "数量不匹配: 标准答案中图片数量为%d, 而检测结果中图片数量为%d" % (
    len(groundtruth), len(results))
    
    maxiou_confidence = np.array([])
    
    for i in range(len(results)):
        
        print(results[i][0])
        
        if show_images: # 若需可视化
            fname = './' + results[i][0] + '.jpg' # 若需可视化, 修改这里为存放图片的路径
            image = cv2.imread(fname)
            
        for j in range(2, len(results[i])): # 对于一张图片中的每一个检测框
            
            iou_array = np.array([])
            detectedbox = results[i][j]
            confidence = detectedbox[-1]
            
            if show_images: # 若需可视化
                x_min, y_min = int(detectedbox[0]), int(detectedbox[1])
                x_max = int(detectedbox[0] + detectedbox[2])
                y_max = int(detectedbox[1] + detectedbox[3])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
            for k in range(2, len(groundtruth[i])): # 去匹配这张图片中的每一个标注框
                groundtruthbox = groundtruth[i][k]
                iou = cal_IoU(detectedbox, groundtruthbox)
                iou_array = np.append(iou_array, iou) # 得到一个交并比的数组
                
                if show_images: # 若需可视化
                    x_min, y_min = int(groundtruthbox[0]), int(groundtruthbox[1])
                    x_max = int(groundtruthbox[0] + groundtruthbox[2])
                    y_max = int(groundtruthbox[1] + groundtruthbox[3])
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
            maxiou = np.max(iou_array) #最大交并比
            maxiou_confidence = np.append(maxiou_confidence, [maxiou, confidence])
            
        if show_images: # 若需可视化
            cv2.imshow("Image",image)
            cv2.waitKey()
            
    maxiou_confidence = maxiou_confidence.reshape(-1, 2)
    maxiou_confidence = maxiou_confidence[np.argsort(-maxiou_confidence[:, 1])] # 按置信度从大到小排序

    return maxiou_confidence, num_detectedbox, num_groundtruthbox


def thres(maxiou_confidence, threshold = 0.5):
    """
    将大于阈值的最大交并比记为1, 反正记为0
    :param maxiou_confidence: np.array, 存放所有检测框对应的最大交并比和置信度
    :param threshold: 阈值
    :return tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
    """
    maxious = maxiou_confidence[:, 0]
    confidences = maxiou_confidence[:, 1]
    true_or_flase = (maxious > threshold)
    tf_confidence = np.array([true_or_flase, confidences])
    tf_confidence = tf_confidence.T
    tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
    return tf_confidence


def plot(tf_confidence, num_groundtruthbox):
    """
    从上到下截取tf_confidence, 计算并画图
    :param tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
    :param num_groundtruthbox: int, 标注框的总数
    """
    fp_list = []
    recall_list = []
    precision_list = []
    auc = 0
    mAP = 0
    for num in range(len(tf_confidence)):
        arr = tf_confidence[:(num + 1), 0] # 截取, 注意要加1
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        recall = tp / num_groundtruthbox
        precision = tp / (tp + fp)
        auc = auc + recall
        mAP = mAP + precision

        fp_list.append(fp)
        recall_list.append(recall)
        precision_list.append(precision)
    
    auc = auc / len(fp_list)
    mAP = mAP * max(recall_list) / len(recall_list)
    
    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    plt.plot(fp_list, recall_list, label = 'AUC: ' + str(auc))
    plt.legend()

    plt.figure()
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.plot(recall_list, precision_list, label = 'mAP: ' + str(mAP))
    plt.legend()
    
    plt.show()


def ellipse_to_rect(ellipse):
    """
    将椭圆框转换为水平竖直的矩形框
    :param ellipse: list, [major_axis_radius minor_axis_radius angle center_x center_y, score]
    :return rect: list, [leftx, topy, width, height, score]
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
    :param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return iou: 交并比
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

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
    读取检测结果或 groundtruth 的文档, 若为椭圆坐标, 转换为矩形坐标
    :param txtfile: 读入的.txt文件, 格式要求与FDDB相同
    :return imagelist: list, 每张图片的信息单独为一行, 第一列是图片名称, 第二列是人脸个数, 后面的列均为列表, 包含4个矩形坐标和1个分数
    :return num_allboxes: int, 矩形框的总个数
    '''
    imagelist = [] # 包含所有图片的信息的列表
    
    txtfile = open(txtfile, 'r')
    lines = txtfile.readlines() # 一次性全部读取, 得到一个list
    
    num_allboxes = 0
    i = 0
    while i < len(lines): # 在lines中循环一遍
        image = [] # 包含一张图片信息的列表
        image.append(lines[i].strip()) # 去掉首尾的空格和换行符, 向image中写入图片名称
        num_faces = int(lines[i + 1])
        num_allboxes = num_allboxes + num_faces
        image.append(num_faces) # 向image中写入人脸个数
        
        if num_faces > 0:
            for num in range(num_faces):
                boundingbox = lines[i + 2 + num].strip() # 去掉首尾的空格和换行符
                boundingbox = boundingbox.split() # 按中间的空格分割成多个元素
                boundingbox = list(map(float, boundingbox)) # 转换成浮点数列表
                
                if len(boundingbox) == 6: # 如果是椭圆坐标
                    boundingbox = ellipse_to_rect(boundingbox) # 则转换为矩形坐标
                    
                image.append(boundingbox) # 向image中写入包含矩形坐标和分数的浮点数列表
                
        imagelist.append(image) # 向imagelist中写入一张图片的信息
        
        i = i + num_faces + 2 # 增加index至下张图片开始的行数
        
    txtfile.close()
    
    return imagelist, num_allboxes