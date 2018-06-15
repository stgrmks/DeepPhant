_author__ = 'MSteger'

import numpy as np
import torch
import PIL
import os, gc
import scipy.ndimage as nd
import PIL.Image
from torch.autograd import Variable
from torchvision import transforms, models
from tqdm import tqdm

class DreamPhant(object):

    def __init__(self, model, input_dir, device = torch.device('cpu'), step_fn=None, verbose = True):
        self.model = model.to(device)
        self.input_dir = input_dir
        self.device = device
        self.step_fn = self.make_step if step_fn is None else step_fn
        self.verbose = verbose

    def _load_image(self, path, preprocess, resize = None):
        img = PIL.Image.open(path)
        if resize is not None: img.thumbnail(resize, PIL.Image.ANTIALIAS)
        img_tensor = preprocess(img).unsqueeze(0) if preprocess is not None else transforms.ToTensor(img)
        return img, img_tensor, img_tensor.numpy()

    def _data_to_img(self, t, tensor = True):
        if tensor: t = t.numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
        std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
        inp = t[0, :, :, :]
        inp = inp.transpose(1, 2, 0)
        inp = std * inp + mean
        inp *= 255
        inp = np.uint8(np.clip(inp, 0, 255))
        return PIL.Image.fromarray(inp)

    def _image_to_variable(self, image, requires_grad=False):
        return Variable(image.cuda() if self.device == torch.device('cuda') else image, requires_grad=requires_grad)

    def _extract_features(self, img_tensor, layer, model = None):
        if model is None: model = self.model
        features = self._image_to_variable(img_tensor, requires_grad=True) if not isinstance(img_tensor, (torch.cuda.FloatTensor if self.device == torch.device('cuda')  else torch.Tensor)) else img_tensor
        for index, current_layer in enumerate(model.features.children()):
            features = current_layer(features)
            if index == layer: break
        return features

    def objective(self, dst, guide_features=None):
        if guide_features is None:
            return dst.data
        else:
            x = dst.data[0].cpu().numpy()
            y = guide_features.data[0].cpu().numpy()
            ch, w, h = x.shape
            x = x.reshape(ch, -1)
            y = y.reshape(ch, -1)
            A = x.T.dot(y)
            diff = y[:, A.argmax(1)]
            return torch.Tensor(np.array([diff.reshape(ch, w, h)])).to(self.device)

    def make_step(self, img, control=None, step_size=1.5, layer=28, jitter=32):

        mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
        std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

        ox, oy = np.random.randint(-jitter, jitter + 1, 2) # offset by random jitter

        img = np.roll(np.roll(img, ox, -1), oy, -2) # apply jitter shift
        tensor = torch.Tensor(img)

        img_var = self._image_to_variable(tensor, requires_grad=True)
        self.model.zero_grad()

        x = self._extract_features(img_tensor=img_var, layer=layer)
        delta = self.objective(x, control)
        x.backward(delta)

        # L2 Regularization on gradients
        mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2)]).to(self.device)
        img_var.grad.data /= torch.sqrt(mean_square)
        img_var.data.add_(img_var.grad.data * step_size)

        result = img_var.data.cpu().numpy()
        result = np.roll(np.roll(result, -ox, -1), -oy, -2)
        result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std) # normalize img

        return torch.Tensor(result)

    def DeepDream(self, base_img, octave_n=6, octave_scale=1.4, iter_n=10, **step_args):
        octaves = [base_img]
        for i in range(octave_n - 1): octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

        detail = np.zeros_like(octaves[-1])

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)
            src = octave_base + detail
            for i in range(iter_n):
                src = self.step_fn(src, **step_args)

            detail = src.numpy() - octave_base

        return src

    def transform(self, preprocess, layer, control=None, resize = [1024, 1024], repeated = 10, file_prefix = None,**dream_args):
        if (repeated is None) | (repeated is False): repeated = 1
        if control is not None:
            _, guideImage_tensor, _ = self._load_image(path=control[2], preprocess=control[3], resize=control[4])
            control = self._extract_features(img_tensor=guideImage_tensor, layer = control[0])
        for img_name in os.listdir(self.input_dir):
            img_path = os.path.join(self.input_dir, img_name)
            _, _, frame = self._load_image(path=img_path, preprocess=preprocess, resize=resize)
            bar = tqdm(total=repeated, unit='iteration')
            for i in range(repeated):
                frame = self.DeepDream(base_img=frame, layer = layer, control = control, **dream_args).numpy()
                bar.update(1)
                bar.set_description('Processing Image: {}'.format(img_name))
            bar.close()
            DeepDream = self._data_to_img(frame, tensor=False)
            output_dir = os.path.join(self.input_dir.replace('/input', '/output'), 'layer{}'.format(layer))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            if file_prefix is not None: img_name = '{}_{}'.format(file_prefix, img_name)
            output_path = os.path.join(output_dir, img_name)
            DeepDream.save(output_path)
            if self.verbose: print 'saved img {} to {}'.format(os.path.split(img_name)[-1], output_path)
        return self

if __name__ == '__main__':
    from utils.helpers import summary
    print torch.__version__

    # setup
    input_dir = r'/media/msteger/storage/resources/DreamPhant/dream/input/'
    guideImage_dir = r'/media/msteger/storage/resources/DreamPhant/dream/guides/selected/'
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    preprocess_resize = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    device = torch.device('cuda')
    iter_n = 5
    step_size = 0.01
    jitter = 32
    guided = True

    # model
    model = models.vgg19(pretrained=True)
    for weights in model.parameters(): weights.requires_grad = False
    summary(model=model, device=device, input_size=(1,3,224,224))

    # dreaming
    for guideImage_name in os.listdir(guideImage_dir):
        guideImage_path = os.path.join(guideImage_dir, guideImage_name)
        guideImage_name = guideImage_name.split('.jpg')[0]
        for rep in range(30, 120, 30):
            Dream = DreamPhant(model=model, input_dir=input_dir, device=device)
            Dream.transform(preprocess = preprocess, resize = [768, 1024], layer = 20, octave_n=6, octave_scale=1.4,iter_n=iter_n, control=(model, 20, guideImage_path, preprocess_resize, None) if guided else None,\
                            step_size=step_size, jitter=jitter, repeated = rep, file_prefix='{}_{}_{}_{}_{}_{}'.format(guideImage_name, rep, iter_n, step_size, jitter, guided))
            Dream = None
            gc.collect()

    print 'done'