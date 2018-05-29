_author__ = 'MSteger'

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.modules.module import _addindent

def mv_file(file_path, rm_folder_in_path = 'images'):
    f = file_path.split('/')
    if rm_folder_in_path in f:
        f.remove(rm_folder_in_path)
    else:
        return
    new_path = os.path.join(*['/'] + f)
    if not os.path.isfile(new_path): os.rename(file_path, new_path)
    return

def torch_summarize(model, show_weights = True, show_parameters = True):
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():

        if type(module) in [torch.nn.modules.container.Container, torch.nn.modules.container.Sequential]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()

        modstr = _addindent(modstr, 2)
        trainable_params, frozen_params, weights = [], [], []
        for p in module.parameters():
            weights.append(tuple(p.size()))
            if p.requires_grad:
                trainable_params.append(np.prod(p.size()))
            else:
                frozen_params.append(np.prod(p.size()))

        trainable_params, frozen_params, weights = sum(trainable_params), sum(frozen_params), tuple(weights)

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', trainable params={}'.format(trainable_params)
            tmpstr +=  ', frozen params={}'.format(frozen_params)

        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

def summary(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('----------------------------------------------------------------')
    line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
    print(line_new)
    print('================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']),
                                                  summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('----------------------------------------------------------------')
    # return summary

if __name__ == '__main__':
    print 'done'