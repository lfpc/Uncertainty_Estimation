"""
Basen on https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""

import numpy as np
from .__init__ import Cifar10,Cifar100

class ImbalanceCifar10(Cifar10):
    def __init__(self, imbalance_ratio = 0.01,imb_type = 'exp', train = True, test = True,
                        apply_test = False,
                        params = Cifar10.params,
                        name='CIFAR 10 - LT',
                        download=True, 
                        data_dir="data"):
        super().__init__(params, name, download, data_dir, train = train, test = test)

        if self.training_data is not None:
            img_num_list = self.get_img_num_per_cls(self.training_data,self.n_classes, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.training_data,img_num_list)
        if self.training_data is not None and apply_test:
            img_num_list = self.get_img_num_per_cls(self.test_data,self.n_classes, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.test_data,img_num_list)
        #self.generate_dataloaders()


    def get_img_num_per_cls(self,data, cls_num, imb_type, imb_factor):
        img_max = len(data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def gen_imbalanced_data(self,dataset, img_num_per_cls):
        data = dataset.data
        targets = dataset.targets
        new_data = []
        new_targets = []
        targets_np = np.array(targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        dataset.data = new_data
        dataset.targets = new_targets
        return dataset

    def get_num_classes(self):
        return self.n_classes

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.n_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class ImbalanceCifar100(Cifar100):
    def __init__(self, imbalance_ratio = 0.01,imb_type = 'exp', train = True, test = True,
                        apply_test = False,
                        params = Cifar100.params,
                        name='CIFAR 100 - LT',
                        download=True, 
                        data_dir="data"):
        super().__init__(params, name, download, data_dir, train = train, test = test)
        img_num_list = self.get_img_num_per_cls(self.n_classes, imb_type, imbalance_ratio)
        if self.training_data is not None:
            self.gen_imbalanced_data(self.training_data,img_num_list)
        if self.training_data is not None and apply_test:
            self.gen_imbalanced_data(self.test_data,img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def gen_imbalanced_data(self,dataset, img_num_per_cls):
        data = dataset.data
        targets = dataset.targets
        new_data = []
        new_targets = []
        targets_np = np.array(targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        dataset.data = new_data
        dataset.targets = new_targets
        return dataset

    def get_num_classes(self):
        return self.n_classes

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.n_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list