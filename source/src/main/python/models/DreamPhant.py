_author__ = 'MSteger'

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
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


def deep_dream_vgg(image, layer, iterations, lr, octave_scale, num_octaves):
    if num_octaves > 0:
        image1 = image.filter(ImageFilter.GaussianBlur(2))
        if (image1.size[0] / octave_scale < 1 or image1.size[1] / octave_scale < 1):
            size = image1.size
        else:
            size = (int(image1.size[0] / octave_scale), int(image1.size[1] / octave_scale))

        image1 = image1.resize(size, Image.ANTIALIAS)
        image1 = deep_dream_vgg(image1, layer, iterations, lr, octave_scale, num_octaves - 1)
        size = (image.size[0], image.size[1])
        image1 = image1.resize(size, Image.ANTIALIAS)
        image = ImageChops.blend(image, image1, 0.6)
        print("-------------- Recursive level: ", num_octaves, '--------------')
    img_result = dd_helper(image, layer, iterations, lr)
    img_result = img_result.resize(image.size)
    plt.imshow(img_result)
    return img_result

if __name__ == '__main__':
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalise
    ])

    model = models.vgg16(pretrained = True).cuda()
    modulelist = list(model.features.modules())

    input_pic = load_image('/home/mks/ownCloud/git/sandbox/data/images/Elephant.jpg')
    output_pic = deep_dream_vgg(input_pic, 28, 5, 0.2, 2, 20)
    print 'done'