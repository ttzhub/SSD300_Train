import numpy as np
import os
from xml.etree import ElementTree

class TransData(object):

    def __init__(self,  data_path, class_name_path):
        self.path_prefix = data_path
        # self.num_classes = num_classes
        self.TD_fn_read_class_name(class_name_path)
        self.data = dict()
        self.Preprocess_XML()

    def Preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)

                # 有后缀的同学使用
                class_name = object_tree.find('name').text

                # 没后缀的同学使用
                # class_name = object_tree.find('name').text + '图片文件的后缀'

                one_hot_class = self.one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            # print('bounding_boxes:{}'.format(bounding_boxes))
            one_hot_classes = np.asarray(one_hot_classes)
            # print('one_hot_classes:{}'.format(one_hot_classes))
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            # print('image_data:{}'.format(image_data))
            self.data[image_name] = image_data

    def one_hot(self, name):
        """
        变为one_hot向量，即类别对应位为1，其他为0
        :param name:
        :return:
        """
        one_hot_vector = [0] * self.num_classes
        try:
            index = self.class_name.index(name)
            one_hot_vector[index] = 1
            return one_hot_vector
        except ValueError as e:
            print("名称文件中没有该类名！！\n", e)

    def TD_fn_read_class_name(self, filePath):
        """
        从TXT文件中读取类别list
        :param filePath: TXT文件路径
        :return:
        """
        self.class_name = []
        try:
            with open(filePath, 'r') as f:
                for line in f:
                    name = line.split()[0]
                    self.class_name.append(name)
        except:
            print('{}打开文件错误！'.format(filePath))
        self.num_classes = len(self.class_name)
        print(self.class_name)



import pickle

def main():
    # 获取标记数据
    # data_path 为标记产生的 .xml 文件所在的文件夹路径
    # class_name_path 为 4.1 中创建的类别名称文件路径
    data = TransData(data_path='E:\my_data\Annotations\\',
                     class_name_path='E:\my_data\\name.txt').data

    # 保存写入文件
    pickle.dump(data, open('my_new_data.pkl', 'wb'))
    print("提取成功！！！！")

if __name__ == '__main__':
    main()
    '''
    gt1 = pickle.load(open('D:\Python Space\SSD\ssd_keras-master(keras2)\PASCAL_VOC\my_data.pkl', 'rb'))
    gt2 = pickle.load(open('D:\Python Space\SSD\ssd_keras-master(keras2)\PASCAL_VOC\my_new_data.pkl', 'rb'))
    key1 = gt1.keys()
    key2 = gt2.keys()
    print("gt1.data\n", gt1['000001.jpg'])
    print("gt1.data\n", gt2['000001.jpg'])
    '''