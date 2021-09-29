import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
from torch.utils import data
from tqdm import tqdm
from ptsemseg.models import get_model
from ptsemseg.utils import get_logger
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict
import cv2
from PIL import Image
from BMPWriter import BMPWriter as bmpw
import pandas
from pdb import set_trace
import openslide

parser = argparse.ArgumentParser(description="Params")
parser.add_argument( "--model_path", nargs="?", type=str, default="models/DMMN-breast.pkl")
parser.add_argument( "--out_path", nargs="?", type=str, default="imgs/DMMN-breast", help="Path of the output segmap")
args = parser.parse_args()

### Breast pretrained model segmentation classes ###
# Class 1 (red): stroma
# Class 2 (blue): nectoric
# Class 3 (yellow): carcinoma
# Class 4 (gray): background
# Class 5 (green): adipose
# Class 6 (orange): benign epithelial

def test():
    bmp = bmpw()
    model_file_name = os.path.split(args.model_path)[1]
    model_name  = {}
    model_name['arch'] = "DMMN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 7 # the number of tissue subtype classes + 1
    test_file = "test_coords.csv" # the list of patch coordinates
    with open(test_file) as f1:
        test_file_patches = [line.rstrip('\n') for line in f1]
    data_path = "testing_images/" # the path where testing whole slide images are located
    tile_size = 1024
    problem_type = 'tissue'
    batch_size = 1
    with open(test_file) as f:
        file_names = [line.rstrip('\n') for line in f]
    
    model = get_model(model_name, n_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.cuda()

    if not os.path.exists(str(args.out_path)):
        os.makedirs(str(args.out_path))

    test_file_patches_split = test_file_patches[0].split(",")
    pslide_id = int(test_file_patches_split[0])
    filename = os.path.join(os.path.abspath(args.out_path) + "/" + str(pslide_id) + ".svs_data/predictions.bmp")
    slide = openslide.OpenSlide(data_path + str(pslide_id) + ".svs")
    tilecnt = 0
    
    with torch.no_grad():
        for ii in tqdm(range(0,len(test_file_patches))):
            test_file_patches_split = test_file_patches[ii].split(",")
            slide_id = int(test_file_patches_split[0])
            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            inputs_slide = slide.read_region((xmin,ymin), 0, (tile_size,tile_size)).convert('RGB')
            inputs_slide = np.array(inputs_slide)/255.0
            inputs_slide = np.expand_dims(np.transpose(inputs_slide, (2,1,0)), axis=0)
            inputs = torch.from_numpy(inputs_slide).float().to(device)
            inputs = torch.flip(inputs.permute(0,1,3,2),[1])
            
            if tilecnt >= len(test_file_patches):
                outfile = color_change(outfile)
                image_size = slide.dimensions
                bmp.writebmp(filename,outfile,int(image_size[0]), int(image_size[1]),palette='standard')
                break

            filename = os.path.join(os.path.abspath(args.out_path) + "/" + str(slide_id) + ".svs_data/predictions.bmp")
            if slide_id != pslide_id:
                pfilename = os.path.join(os.path.abspath(args.out_path) + "/" + str(pslide_id) + ".svs_data/predictions.bmp")
                outfile = color_change(outfile)
                image_size = slide.dimensions
                bmp.writebmp(pfilename,outfile,int(image_size[0]), int(image_size[1]),palette='standard')
                slide = openslide.OpenSlide(data_path + str(slide_id) + ".svs")
            if not os.path.isdir(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image_size = slide.dimensions
                outfile = bmp.makeempty(int(image_size[0]), int(image_size[1]))

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if xmin == 0 and ymin == 0:
                # tile at outfile[0:128,0:128]
                ref_pad = nn.ReflectionPad2d(512)
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[:128,:64] = t_mask_shrink[128:,64:]

                # tile at outfile[0:128,128:384]
                ref_pad = nn.ReflectionPad2d((512,512,256,256))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[:128,64:192] = t_mask_shrink[128:,:]

                # tile at outfile[128:384,0:128]
                ref_pad = nn.ReflectionPad2d((256,256,512,512))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[128:384,:64] = t_mask_shrink[:,64:]

                # tile at outfile[128:384,128:384]
                ref_pad = nn.ReflectionPad2d(256)
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[128:384,64:192] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if xmin == 0:
                ymin += 384
                # tile at outfile[0:128,ymin:ymin+256]
                ref_pad = nn.ReflectionPad2d((512,512,0,0))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t2 = int(ymin)
                wh2 = int(image_size[1])
                if t2 < wh2:
                    if t2 + 256 > wh2:
                        outfile[t2:,:64] = t_mask_shrink[0:wh2-t2,64:]
                    else:
                        outfile[t2:t2+256,:64] = t_mask_shrink[:,64:]

                # tile at outfile[128:384,ymin:ymin+256]
                ref_pad = nn.ReflectionPad2d((256,256,0,0))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t2 = int(ymin)
                wh2 = int(image_size[1])
                if t2 < wh2:
                    if t2 + 256 > wh2:
                        outfile[t2:,64:192] = t_mask_shrink[0:wh2-t2,:]
                    else:
                        outfile[t2:t2+256,64:192] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if ymin == 0:
                xmin += 384
                # tile at outfile[xmin:xmin+256,0:128]
                ref_pad = nn.ReflectionPad2d((0,0,512,512))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t1 = bmp.getrowsize(int(xmin))
                wh1 = bmp.getrowsize(int(image_size[0]))
                if t1 < wh1:
                    if t1 + 128 > wh1:
                        outfile[:128,t1:] = t_mask_shrink[128:,0:wh1-t1]
                    else:
                        outfile[:128,t1:t1+128] = t_mask_shrink[128:,:]

                # tile at outfile[xmin:xmin+256,128:384]
                ref_pad = nn.ReflectionPad2d((0,0,256,256))
                inputs_new = torch.zeros([1,3,1024,1024]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024,:1024]
                outputs = model(inputs_new[:,:,384:640,384:640],inputs_new[:,:,::2,::2][:,:,128:384,128:384],inputs_new[:,:,::4,::4])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t1 = bmp.getrowsize(int(xmin))
                wh1 = bmp.getrowsize(int(image_size[0]))
                if t1 < wh1:
                    if t1 + 128 > wh1:
                        outfile[128:384,t1:] = t_mask_shrink[:,0:wh1-t1]
                    else:
                        outfile[128:384,t1:t1+128] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            outputs = model(inputs[:,:,384:640,384:640],inputs[:,:,::2,::2][:,:,128:384,128:384],inputs[:,:,::4,::4])
            t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
            t_mask = t_masks[0,:,:]
            xmin += 384
            ymin += 384
            t_mask_shrink = mask_shrink(t_mask)
            t1 = bmp.getrowsize(int(xmin))
            t2 = int(ymin)
            image_size = slide.dimensions
            wh1 = bmp.getrowsize(int(image_size[0]))
            wh2 = int(image_size[1])
            if t1 < wh1 and t2 < wh2:
                if t2 + 256 > wh2 and t1 + 128 > wh1:
                    outfile[t2:,int(int(xmin)/2):] = t_mask_shrink[0:wh2-t2,0:wh1-t1]
                elif t2 + 256 > wh2:
                    outfile[t2:,t1:t1+128] = t_mask_shrink[0:wh2-t2,:]
                elif t1 + 128 > wh1:
                    outfile[t2:t2+256,t1:] = t_mask_shrink[:,0:wh1-t1]
                else:
                    outfile[t2:t2+256,t1:t1+128] = t_mask_shrink
            pslide_id = slide_id
            tilecnt += 1
    outfile = color_change(outfile)
    image_size = slide.dimensions
    bmp.writebmp(filename,outfile,int(image_size[0]), int(image_size[1]),palette='standard')

def color_change(t_mask):
    t_mask[t_mask == 0] = 4*17 # label excluded regions by Otsu algorithm as background (class 4)
    return t_mask

def mask_shrink(t_mask):
    t_mask_shrink = t_mask[:,::2]*17
    return t_mask_shrink


if __name__ == "__main__":
    test()
