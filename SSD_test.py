import cv2
import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from rely.ssd_utils import BBoxUtility
from rely.ssd_v2 import SSD300v2

STANDARD_COLORS = [
    (100, 30, 22), (120, 40, 31), (81, 46, 95), (74, 35, 90), (21, 67, 96), (27, 79, 114), (14, 98, 81),
    (11, 83, 69), (20, 90, 50), (24, 106, 59), (125, 102, 8), (126, 81, 9), (120, 66, 18), (110, 44, 0),
    (66, 73, 73), (27, 38, 49), (192, 57, 43), (231, 76, 60), (155, 89, 182), (142, 68, 173), (41, 128, 185),
    (52, 152, 219), (26, 188, 156), (22, 160, 133), (39, 174, 96), (46, 204, 113), (243, 156, 18), (230, 126, 34),
    (211, 84, 0), (127, 140, 141), (39, 55, 70)
]

def color_gen():
    """
    颜色生成器
    :return:
    """
    global STANDARD_COLORS
    while True:
        # 随机排序
        np.random.shuffle(STANDARD_COLORS)
        for color in STANDARD_COLORS:
            yield color

class SSD_test(object):

    def __init__(self, weight_path, class_nam_list):
        self.input_shape = (300, 300, 3)
        self.voc_classes = class_nam_list
        self.NUM_CLASSES = len(self.voc_classes) + 1
        self.weight_path = weight_path
        self.bbox_util = BBoxUtility(self.NUM_CLASSES)

        self.BuildSSD()

    # 建立模型
    def BuildSSD(self):
        """
        建立模型并载入权值文件
        :return:
        """
        self.model = SSD300v2(self.input_shape, num_classes=self.NUM_CLASSES)
        self.model.load_weights(self.weight_path, by_name=True)


    # 模型预测
    def Predict(self, img, min_score=0.6):
        """
        预测Img
        :param img: 带检测图片
        :param min_score: 阈值，过滤置信度小于其值的目标
        :return:预测结果
        """
        inputs = cv2.resize(img, (300, 300))
        inputs = image.img_to_array(inputs)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = preprocess_input(inputs)
        preds = self.model.predict(inputs, batch_size=1, verbose=0)  # verbose = 1 显示耗时
        results = self.bbox_util.detection_out(preds)  # 非最大抑制
        h, w = img.shape[:2]
        preds = []
        det_label = results[0][:, 0]  # 类别索引
        det_conf = results[0][:, 1]  # 概率
        det_xmin = results[0][:, 2]  # 坐标
        det_ymin = results[0][:, 3]  # 坐标
        det_xmax = results[0][:, 4]  # 坐标
        det_ymax = results[0][:, 5]  # 坐标

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= min_score]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]

            score = top_conf[i]
            label = int(top_label_indices[i]) - 1
            preds.append((label, score, xmin, ymin, xmax, ymax))
        return preds


    def filter(self, preds, inclued_class=[]):
        """
        过滤器，过滤掉不在inclued_class里的预测结果
        :param preds: 预测结果
        :param inclued_class: 需要保留的类名
        :return:
        """
        out = []
        for lab, score, xmin, ymin, xmax, ymax in preds:
            if self.voc_classes[lab] in inclued_class:
                out.append((self.voc_classes[lab], score, xmin, ymin, xmax, ymax))
        return out

    def draw_img(self, img, preds):
        """
        绘制预测结果
        :param img: 源图片
        :param preds: 预测结果
        :return: 绘制目标框的Img
        """
        h, w = img.shape[:2]
        offset = round(h * 0.02)
        text_height = (h * 0.0015)
        line_thickness = round(h * 0.005)
        text_thickness = round(h * 0.004)
        gen_color = color_gen()
        for i, pred in enumerate(preds):
            lab, score, xmin, ymin, xmax, ymax = pred
            text = lab + ' {:.3f}'.format(score)
            xmin = int(round(xmin * w))
            ymin = int(round(ymin * h))
            xmax = int(round(xmax * w))
            ymax = int(round(ymax * h))
            if ymin - offset <= 0:
                T_x = xmin + offset
                T_y = ymin + round(2.5*offset)
            else:
                T_x = xmin + offset
                T_y = ymin - offset
            color = gen_color.__next__()
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.putText(img, text, (T_x, T_y), cv2.FONT_HERSHEY_SIMPLEX, text_height,
                        color, text_thickness)
        return img
            


if __name__ == '__main__':
    import os
    import sys
    work_space = os.path.split(sys.argv[0])[0]
    os.chdir(work_space)

    weight_path = './weights/weights_SSD300.hdf5'
    class_nam_list = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    ssd = SSD_test(weight_path, class_nam_list)

    img = cv2.imread('test2.jpg', )
    # img = cv2.imread('test.jpg', )
    # img = cv2.imread('fishbike.jpg', )

    img = cv2.resize(img, (720, 480))
    pred = ssd.Predict(img, 0.6)
    # print(pred)
    pred = ssd.filter(pred, class_nam_list)
    img = ssd.draw_img(img, pred)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()