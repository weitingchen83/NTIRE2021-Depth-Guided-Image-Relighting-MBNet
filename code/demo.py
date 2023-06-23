import os
import glob
import torch
import torch.nn
import torch.utils.data
import torchvision
from PIL import Image

from config_val import get_config
import numpy as np
import hdfnet
import math


def make_test_data(cfg, img_path_list, device, depth=True):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

   ])
    imgs = []
    depths = []
    for img_path in img_path_list:
        if img_path.endswith('.npy'):
            continue
        x = data_transform(Image.open(str(img_path))).unsqueeze(0)
        x = x.to(device)
        x = x[:,:3,:,:]

        if depth:

            seg_depth_name = img_path.replace('.png','.npy')
            seg_depth = np.load(seg_depth_name, allow_pickle=True).item()['normalized_depth']
            seg_depth = torch.unsqueeze(torch.from_numpy(seg_depth), 0)
            seg_depth =  torch.unsqueeze(seg_depth, 0)
            depths.append(seg_depth)

        imgs.append(x)
    
    if depth:
        return imgs, depths
    return imgs


def build_network(model_name):
    return hdfnet.HDFNet_Res50().cuda()

    

def load_pretrain_network(model_name, model_path):

    network = build_network(model_name)
    network.load_state_dict(torch.load(model_path)['state_dict'])
    network.eval().cpu()
    return network


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------

    path = cfg.input_data_path
    if cfg.gt:
        gt_path = cfg.ori_data_path
    output_dir = cfg.output_dir


    name = os.listdir(path)
    print('Start eval')

    networks_description = [[cfg.type, cfg.model_path]]

    networks = [load_pretrain_network(network[0], network[1]) for network in networks_description]
    
    if not os.path.exists(os.path.join(output_dir, cfg.name)):
        os.makedirs(os.path.join(output_dir, cfg.name))
    
    if cfg.gt:
        psrn, ssim = [], []

 

    for i in name:

        if i.endswith('.npy'):
            continue

        test_file_path = os.path.join(path, i)
        test_file_path = glob.glob(test_file_path)
        test_images, test_depths = make_test_data(cfg, test_file_path, device)

        test_img, test_depth = test_images[0], test_depths[0]

        for idx, network in enumerate(networks):
            
            network.cuda()
            
            if idx == 0:

                dehaze_image = network(test_img.cuda(), test_depth.cuda()).detach()
                dehaze_image = dehaze_image.cpu()
            else:
                tmp =  network(test_img.cuda(), test_depth.cuda()).detach()
                dehaze_image += tmp.cpu()
            
            network.cpu()
            

        dehaze_image = dehaze_image/(len(networks))
        dehaze_image = dehaze_image.cuda()

        torchvision.utils.save_image(dehaze_image, os.path.join(output_dir, cfg.name, i))
        
        del dehaze_image
    
    if cfg.gt:
        print(sum(psrn)/len(psrn), sum(ssim)/len(ssim))




if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)