import torch
import time
from pdb import set_trace
import os
import numpy as np
import openslide
import extract_tissue
import shutil
import math
import glob
from slidereader_interface import SlideReaderCoords

if __name__ == '__main__':
    
    slides_to_read = ["testing_images/1.svs","testing_images/2.svs","testing_images/3.svs"] # list of testing whole slide images
    coord_file = open('test_coords.csv', 'w') # a file listing all patch coordinates

    ### get coordinates of tissue patches in whole slide images using Otsu algorithm ###
    for image_path in slides_to_read:
        slide = openslide.OpenSlide(image_path)
        slide_id = os.path.splitext(os.path.basename(image_path))[0]
        grid, _ = extract_tissue.make_sample_grid(slide, 256,
               20, 10, 10, False, prune=False, overlap=0)
        for (x,y) in grid:
            coord_file.write('{},{},{},0\n'.format(slide_id, x, y))

    ### get coordinates of all patches in whole slide images ###
    # stride = 256
    # for image_path in slides_to_read:
    #     slide = openslide.OpenSlide(image_path)
    #     slide_id = os.path.splitext(os.path.basename(image_path))[0]
    #     image_size = slide.dimensions
    #     for ii in range(0,image_size[0],stride):
    #         for jj in range(0,image_size[1],stride):
    #             coord_file.write('{},{},{},0\n'.format(slide_id, ii, jj))