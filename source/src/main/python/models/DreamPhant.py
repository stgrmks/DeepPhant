_author__ = 'MSteger'

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageFilter, ImageChops

def load_image(path):
    img = Image.open(path)
    plt.imshow(img)
    plt.title('image loaded!')
    return img

def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225]).cuda()  + torch.Tensor([0.485, 0.456, 0.406]).cuda()


def dd_helper(image, layer, iterations, lr):
    input = Variable(preprocess(image).unsqueeze(0).cuda(), requires_grad=True)
    model.zero_grad()
    for i in range(iterations):
        print('Iteration: ', i)
        out = input
        for j in range(layer):
            out = modulelist[j + 1](out)
        loss = out.norm()
        loss.backward()
        input.data = input.data + lr * input.grad.data

    input = input.data.squeeze()
    input.transpose_(0, 1)
    input.transpose_(1, 2)
    input = np.clip(deprocess(input), 0, 1)
    im = Image.fromarray(np.uint8(input * 255))
    return im


def deep_dream(image, layer, iterations, lr, octave_scale, num_octaves):
    if num_octaves > 0:
        image1 = image.filter(ImageFilter.GaussianBlur(2))
        if (image1.size[0] / octave_scale < 1 or image1.size[1] / octave_scale < 1):
            size = image1.size
        else:
            size = (int(image1.size[0] / octave_scale), int(image1.size[1] / octave_scale))

        image1 = image1.resize(size, Image.ANTIALIAS)
        image1 = deep_dream(image1, layer, iterations, lr, octave_scale, num_octaves - 1)
        size = (image.size[0], image.size[1])
        image1 = image1.resize(size, Image.ANTIALIAS)
        image = ImageChops.blend(image, image1, 0.6)
        print("-------------- Recursive level: ", num_octaves, '--------------')
    img_result = dd_helper(image, layer, iterations, lr)
    img_result = img_result.resize(image.size)
    plt.imshow(img_result)
    return img_result

if __name__ == '__main__':
    from PhantNet import PhantNet
    chkp_path = r'/media/msteger/storage/resources/DreamPhant/models/run/2018-06-05 20:35:22.740193__0.359831720591__449.pkl'
    input_pic = load_image('/media/msteger/storage/resources/DreamPhant/dream/baby_mother_phants.jpg')
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalise])

    model = PhantNet(input_shape = (3, 224, 224), num_class = 2).cuda() #models.vgg16(pretrained = True).cuda()
    chkp_dict = torch.load(chkp_path)
    model.load_state_dict(chkp_dict['state_dict'])
    modulelist = list(model.features.modules())

    output_pic = deep_dream(image = input_pic, layer = 28, iterations = 5, lr = 0.2, octave_scale = 2, num_octaves = 20)
    print 'done'