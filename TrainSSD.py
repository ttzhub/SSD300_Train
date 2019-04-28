# 依赖文件:     ssd_v2.py    ssd_utils.py
#               weights.02-3.07.hdf5

import cv2
import numpy as np
import pickle
import keras

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from rely.ssd_training import MultiboxLoss

from rely.ssd_v2 import SSD300v2
from rely.ssd_utils import BBoxUtility
from rely.generate import Generator


class TrainSSD(object):

    def __init__(self, weights_path='', Label_Data_path='', ImgDir_path='', PriorBoxes_path='', classes_name=None):
        # 模型建立初始化
        self.voc_classes = classes_name
        self.NUM_CLASSES = len(self.voc_classes) + 1
        self.input_shape = (300, 300, 3)
        self.weights_path = weights_path
        self.bbox_util = BBoxUtility(self.NUM_CLASSES)

        # 模型训练初始化
        self.XML_Data_path = Label_Data_path      # 前处理文件路径
        self.ImgDir_path = ImgDir_path          # 图片集路径
        self.PriorBoxes_path = PriorBoxes_path  # 预设边框文件路径

    # 建立模型
    def SD_fn_BuildSSD(self):
        self.model = SSD300v2(self.input_shape, num_classes=self.NUM_CLASSES)
        self.model.load_weights(self.weights_path, by_name=True)

    # 训练模型
    def SD_fn_TrainSSD(self):
        # 训练准备
        priors = pickle.load(open(self.PriorBoxes_path, 'rb'))
        bbox_util = BBoxUtility(self.NUM_CLASSES, priors)
        gt = pickle.load(open(self.XML_Data_path, 'rb'))
        keys = sorted(gt.keys())
        num_train = int(round(0.8 * len(keys)))
        train_keys = keys[:num_train]
        val_keys = keys[num_train:]
        num_val = len(val_keys)
        # 实例化训练数据迭代器
        gen = Generator(gt=gt,
                        bbox_util=bbox_util,
                        batch_size=16,
                        path_prefix=self.ImgDir_path,
                        train_keys=train_keys, val_keys=val_keys,
                        image_size=(self.input_shape[0], self.input_shape[1]),
                        do_crop=False)

        # 冻结相关层
        '''
        freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                  'conv2_1', 'conv2_2', 'pool2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'pool3']  # ,
        #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']
        '''

        freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                  'conv2_1', 'conv2_2', 'pool2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                  'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv4_3_norm',
                  'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7',
                  'conv6_1', 'conv6_2',
                  'conv7_1', 'conv7_1z', 'conv7_2',
                  'conv8_1', 'conv8_2',
                  'pool6'
                  ]
        for L in self.model.layers:
            if L.name in freeze:
                L.trainable = False


        self.base_lr = 3e-4                     # 定义学习率
        # 定义回合
        callbacks = [keras.callbacks.ModelCheckpoint('.weights/weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                     verbose=1,
                                                     save_weights_only=True),
                     keras.callbacks.LearningRateScheduler(self.schedule)]
        # 配置训练

        optim = keras.optimizers.Adam(lr=self.base_lr)
        self.model.compile(optimizer=optim,
                           loss=MultiboxLoss(self.NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

        nb_epoch = 100  # 训练回合数
        history = self.model.fit_generator(gen.generate(True),              # 训练数据迭代器
                                      gen.train_batches,                    # 一个回合 epoch 中的步数
                                      nb_epoch,                             # 训练回合数
                                      verbose=1,                            # 为1表示不在标准输出流中输出日志信息
                                                                            # #为1表示数据条标准输出显示训练进度
                                                                            # #为2表示每个回合结束后输出一次训练进度
                                      callbacks=callbacks,                  # 回调函数
                                      validation_data=gen.generate(False),  # 验证集
                                      nb_val_samples=gen.val_batches,
                                      nb_worker=1)

    def schedule(self, epoch, decay=0.9):
        return self.base_lr * decay ** (epoch)





    # 模型预测
    def SD_fn_Predict(self, img, min_score=0.6):
        inputs = cv2.resize(img, (300, 300))
        inputs = image.img_to_array(inputs)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = preprocess_input(inputs)
        preds = self.model.predict(inputs, batch_size=1, verbose=0)      # verbose = 1 显示耗时
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
            xmin = int(round(top_xmin[i] * w))
            ymin = int(round(top_ymin[i] * h))
            xmax = int(round(top_xmax[i] * w))
            ymax = int(round(top_ymax[i] * h))

            score = top_conf[i]
            label = int(top_label_indices[i]) - 1
            preds.append((label, score, xmin, ymin, xmax, ymax))
        return preds


    def GetPosition(self, preds):
        pos = []
        for lab, score, xmin, ymin, xmax, ymax in preds:
            if self.voc_classes[lab] == 'Target':
                pos.append((xmin, ymin, xmax, ymax))
        return pos

if __name__ == '__main__':
    import os, sys
    work_space = os.path.split(sys.argv[0])[0]
    os.chdir(work_space)

    name_calss = ['Target', 'person', 'cup', 'fan']

    my_ssd = TrainSSD(weights_path='./weight/weights_SSD300.hdf5',
                      Label_Data_path='my_new_data.pkl',
                      ImgDir_path='D:\my_data\JPEGImages\\',
                      PriorBoxes_path='prior_boxes_ssd300.pkl',
                      classes_name=name_calss)

    my_ssd.SD_fn_BuildSSD()
    my_ssd.SD_fn_TrainSSD()
