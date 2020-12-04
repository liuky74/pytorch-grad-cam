import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ 这实质上是一个包装器,包装了传入的module对象,在调用call函数时除了正常的逐层前向传播外
    还会勾出指定层的梯度,同时返回指定层的输出和整体block的输出"""

    def __init__(self, model,  # 参与热力图计算的block,torch.nn.Module类型
                 target_layers # 具体的层名
                 ):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):  # 导数钩,保存中间运算时该层的梯度
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():  # 循环执行block中的层,勾出的指定层的梯度
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x  # 返回指定层的输出和整个block的输出


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model,  # 整个模型
                 feature_module,  # 进行热力图计算的block,注意这个对象是torch.nn.Module类型的
                 target_layers  # 计算出热力图的具体层名, 字符串类型
                 ):
        self.model = model
        self.feature_module = feature_module
        # 传入block对象,建立钩子获取target_layers指定层的梯度
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):#获取指定层的梯度
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:  # 当执行到指定block时调用包装类来勾出导数
                target_activations, x = self.feature_extractor(x) #取出指定层的输出以及module本身的输出x
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x

# 预处理,包含归一化和permute等
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # 热力图化
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)  # 与原图叠底
    cam = cam / np.max(cam)  # 叠底后归一化
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model,  # 完整的模型
                 feature_module,  # 需要查看热力图的block,这个block中可能包含数个功能层
                 target_layer_names,  # 具体用于生成热力图的层名
                 use_cuda):
        self.model = model #需要查看热力图的模型
        self.feature_module = feature_module # 需要参看热力图的子模块,target_layer_names会指定子模块中的具体层
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        # ModelOutputs会对模型进行包装,通过添加钩子和缓存对象保存指定层的梯度和输出
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)  # 利用包装类的功能获取进行热力图计算的层输出,同时extractor中会使用钩子勾出该层的梯度并缓存
        # 没有指定取哪一类时默认取分类值最高的那类
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        # 独热化,建立一个mask取对应类的logit
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:  # 取出对应类的logit
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        # 只对指定block进行导数归零,开始反向求导
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)  # 反向求导的过程中self.extractor使用钩子会将导数保留
        # 取出指定层的导数,shape = [batch_size,C,H,W],注意这个导数是对index类的导数
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # 取出指定层输出(即feature map)
        target = features[-1]
        # target shape = [bs,C,H,W]
        target = target.cpu().data.numpy()[0, :]
        # 计算每个channel下的导数均值,即每个channel对index类的梯度,shape = [C,]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        """
        将指定层feature map逐个channel乘上梯度均值后求和,根据链式法则,diff*input即为Δw,因此这一步得到的结果可以
        理解为对index这个分类,权重w的梯度在feature map上每个像素的分布,值越大代表该像素对index分类的影响力越大,
        即对于index这个分类,模型应该关注那些像素,即注意力热力图.
        """
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)#类似clip操作,小于0的值全部变为0
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, # 需要提取的模型
                       feature_module=model.layer4,# 提取的特征层
                       target_layer_names=["2"],
                       use_cuda=args.use_cuda)


    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)