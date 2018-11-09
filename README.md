&emsp;&emsp;在评价一个检测模型时通常需要绘制出其 ROC 曲线或 PR 曲线。本程序利用 Python 实现了 ROC 和 PR 曲线的绘制，在 draw_curves 函数中读入 .txt 文件即可一次性绘制出两条曲线并输出 AUC 和 mAP 值，适用于目标检测任务，例如人脸检测。

## 1 流程

为目标检测任务绘制曲线的流程如下：
1. **以检测结果中每一个的 boundingbox 为对象**(记检测出的 boundingbox 的个数为 M)，去匹配该张图片里的每一个 groundtruth boundingbox，计算出交并比 (IoU)，并保留其中最大的值—— maxIoU，同时记录下 confidence 分数。就得到了一个数组—— **maxIoU_confidence，其长度等于 M，宽度为 2**，再按照 confidence 从大到小排序。
2. 设置一个阈值，一般取 0.5。当 maxIoU 大于阈值时，记为 1，即 true positive；当 maxIoU 小于阈值时，记为 0，即 false positve。这样就得到了 **tf_confidence**，其尺寸不变，与 maxIoU_confidence 相同。
3. **从上到下截取数组 tf_confidence 的前 1，2，3，…，M 行**，每次截取都得到一个子数组，子数组中 1 的个数即为 tp，0 的个数即为 fp，查全率 recall (或 TPR) = tp / (groundtruth boundingbox 的个数)，查准率 precision = tp / (tp + fp)。**每次截取得到一个点**，这样就一共得到 M 个点。以 fp 为横坐标，TPR 为纵坐标绘制出 ROC 曲线；以 recall 为横坐标，precision 为纵坐标绘制出 PR 曲线。

## 2 输入

&emsp;&emsp;本程序需要读入两个分别记录检测结果和标准答案的 .txt 文件，记录格式与 FDDB 的要求相同，即

`... `

`image name i `

`number of faces in this image =im `

`face i1 `

`face i2 `

`... `

`face im `

`... `

当检测框为矩形时，face i 为`左上角x 左上角y 宽 高 分数`  
当检测框为椭圆时，格式需要为`长轴半径 短轴半径 角度 中心点x 中心点y 分数`

## 3 输出

&emsp;&emsp;利用本程序在 FDDB 上测试了一个人脸检测模型，绘制出的 ROC 曲线和 PR 曲线如图所示。可以看出此模型误检率很低，但召回率不够高。

![ROC](https://raw.githubusercontent.com/Xingyb14/My_image_hosting_site/master/mtcnn_roc.png)

![PR](https://raw.githubusercontent.com/Xingyb14/My_image_hosting_site/master/mtcnn_pr.png)

[更多说明](https://blog.csdn.net/Xingyb14/article/details/81434087)
