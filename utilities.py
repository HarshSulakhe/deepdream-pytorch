import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from PIL import Image, ImageFilter, ImageChops
import matplotlib.pyplot as plt

import requests
from io import BytesIO

denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                              transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                              ])

class Hook():
    def __init__(self,module,backward = False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self,module,input,output):
        self.input = input
        self.output = output
        # print(type(output))

    def close(self):
        self.hook.remove()

def gradients(net_in, net, layer):
    net_in = net_in.unsqueeze(0).cuda()
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    return net_in.grad.data.squeeze()

def dream(image,model,layer,iterations,lr):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).cuda()
    for i in range(iterations):
        g = gradients(image_tensor,model,layer)
        image_tensor.data = image_tensor.data + lr*g.data

    img_out = image_tensor.detach().cpu()
    img_out = denorm(img_out)
    img_out_np = img_out.numpy().transpose(1,2,0)
    img_out_np = np.clip(img_out_np, 0, 1)
    img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
    return img_out_pil
