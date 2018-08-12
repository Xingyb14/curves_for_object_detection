# coding: utf-8
from Calculate import match, thres, plot


def draw_curves(resultsfile, groundtruthfile, show_images = False, threshold = 0.5):
    """
    读取包含检测结果和标准答案的两个.txt文件, 画出ROC曲线和PR曲线
    :param resultsfile: 包含检测结果的.txt文件
    :param groundtruthfile: 包含标准答案的.txt文件
    :param show_images: 是否显示图片, 若需可视化, 需修改Calculate.match中的代码, 找到存放图片的路径
    :param threshold: IoU阈值
    """
    maxiou_confidence, num_detectedbox, num_groundtruthbox = match(resultsfile, groundtruthfile, show_images)
    tf_confidence = thres(maxiou_confidence, threshold)
    plot(tf_confidence, num_groundtruthbox)


draw_curves("results.txt", "ellipseList.txt")