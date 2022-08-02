import math
import numpy as np
import cv2
import os
import yaml


with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

path_to_fig = config['testing']['fig_path']


def save_as_img(prop, vid_name, branch):
    prop = prop*255
    
    if branch =="bottom":
        save_path = os.path.join(path_to_fig,"GSM_PIC_BOTTOM")
        cv2.imwrite(os.path.join(save_path,vid_name+".png"), prop)
    else:
        save_path = os.path.join(path_to_fig,"GSM_PIC_TOP")
        cv2.imwrite(os.path.join(save_path,vid_name+".png"), prop)


def save_as_img_feat(im, layer):
    im = im*255
    im = np.stack((im,)*3, axis=-1).astype(np.uint8)
    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    
    save_path = os.path.join(path_to_fig,"GSM_PIC_FEAT")
    if layer=="before":

        cv2.imwrite(os.path.join(save_path,"FEAT_BEFORE_.png"), im)
    else:
        cv2.imwrite(os.path.join(save_path,"FEAT_AFTER_.png"), im)
    

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
