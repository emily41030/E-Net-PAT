# import tensorflow as tf

# img1 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [
#                    [1], [2], [3], [4]], [[1], [2], [3], [4]]]], dtype=tf.float32)
# img2 = tf.constant(value=[[[[1], [1], [1], [1]], [[1], [1], [1], [1]], [
#                    [1], [1], [1], [1]], [[1], [1], [1], [1]]]], dtype=tf.float32)
# img = tf.concat(values=[img1, img2], axis=3)
# sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
# print("out1=", type(img))

# # 转化为numpy数组
# img_numpy = img.eval(session=sess)
# print("out2=", type(img_numpy))
# temp = type(img_numpy)
# # 转化为tensor
# img_tensor = tf.convert_to_tensor(img_numpy)
# print("out2=", type(img_tensor))

import torch
import torch.nn as nn
import torchvision.models as models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        for x in range(0, 1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        for i in range(self.slice2.__len__()):
            self.slice2[i].inplace = False
        for i in range(self.slice3.__len__()):
            self.slice3[i].inplace = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        # h = self.slice4(h)
        # h_relu4_3 = h
        # vgg_outputs = namedtuple(
        #     "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)
        return out


# self.VGG_m2_model = VGG_m2(True)
# vgg19 = models.vgg19(pretrained=True)
# pretrained_dict = vgg19.state_dict()

# # 1. filter out unnecessary keys
# m2_pretrained_dict = {k: v for k,
#                       v in pretrained_dict.items() if k in m2_model_dict}

# # 2. overwrite entries in the existing state dict
# m2_model_dict.update(m2_pretrained_dict)

# # 3. load the new state dict
# self.VGG_m2_model.load_state_dict(m2_model_dict)
Vgg191 = Vgg19()
