import cv2
import numpy as np
import openslide
import skimage.morphology
import PIL.Image as Image

def find_level(slide,res,maxres=20.):
    maxres = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    downsample = maxres/res
    for i in range(slide.level_count)[::-1]:
        if slide.level_downsamples[i] <= (downsample+downsample*0.0001):
            level = i
            mult = downsample / slide.level_downsamples[level]
            break
    return level, mult

def image2array(img):
    if img.__class__.__name__=='Image':
        if img.mode=='RGB':
            img=np.array(img)
            r,g,b = np.rollaxis(img, axis=-1)
            img=np.stack([r,g,b],axis=-1)
        elif img.mode=='RGBA':
            img=np.array(img)
            r,g,b,a = np.rollaxis(img, axis=-1)
            img=np.stack([r,g,b],axis=-1)
        else:
            sys.exit('Error: image is not RGB slide')
    img=np.uint8(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def is_sample(img,threshold=0.9,ratioCenter=0.1,wholeAreaCutoff=0.5,centerAreaCutoff=0.9):
    nrows,ncols=img.shape
    timg=cv2.threshold(img, 255*threshold, 1, cv2.THRESH_BINARY_INV)
    kernel=np.ones((5,5),np.uint8)
    cimg=cv2.morphologyEx(timg[1], cv2.MORPH_CLOSE, kernel)
    crow=np.rint(nrows/2).astype(int)
    ccol=np.rint(ncols/2).astype(int)
    drow=np.rint(nrows*ratioCenter/2).astype(int)
    dcol=np.rint(ncols*ratioCenter/2).astype(int)
    centerw=cimg[crow-drow:crow+drow,ccol-dcol:ccol+dcol]
    if (np.count_nonzero(cimg)<nrows*ncols*wholeAreaCutoff) & (np.count_nonzero(centerw)<4*drow*dcol*centerAreaCutoff):
        return False
    else:
        return True

def threshold(slide,size,res,maxres):
    w = int(np.round(slide.dimensions[0]*1./size*res/maxres))
    h = int(np.round(slide.dimensions[1]*1./size*res/maxres))
    thumbnail = slide.get_thumbnail((w,h))
    thumbnail = thumbnail.resize((w,h))
    img = image2array(thumbnail)
    ## remove black dots ##
    _,tmp = cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    tmp = cv2.dilate(tmp,kernel,iterations = 1)
    img[tmp==255] = 255
    #######################
    ## remove 2 pixels around image ##
    if res>10:
        tmp = np.copy(img[2:-2,2:-2])
        img.fill(255)
        img[2:-2,2:-2] = tmp
    ##################################
    img = cv2.GaussianBlur(img,(5,5),0)
    t,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return 255-img,t

def filter_regions(img,min_size,max_ratio):
    l,n = skimage.morphology.label(img, return_num=True)
    for i in range(1,n+1):
        #filter small regions
        if l[l==i].size<min_size:
            l[l==i]=0
#        else:
#            #filter size ratio
#            where = np.where(l==i)
#            ratio = np.ptp(where[0]).astype(float)/np.ptp(where[1])
#            if (ratio>max_ratio)|(1/ratio>max_ratio):
#                l[l==i]=0
    return l

def add(overlap):
    return np.linspace(0,1,overlap+1)[1:-1]

def make_sample_grid(slide,patch_size=224,res=20.,min_cc_size=10,max_ratio_size=10,erode=False,prune=False,overlap=1):
    '''Script that given an openslide object return a list of tuples
    in the form of (x,y) coordinates for patch extraction of sample patches.
    It has an erode option to make sure to get patches that are full of tissue.
    It has a prune option to check if patches are sample. It is slow.'''
    maxres = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    img,th = threshold(slide,patch_size,res,maxres)
    img = filter_regions(img,min_cc_size,max_ratio_size)
    img[img>0]=1
    img = skimage.morphology.binary_dilation(img)
    img = skimage.morphology.binary_dilation(img)
    img = skimage.morphology.binary_dilation(img)
    img = skimage.morphology.binary_dilation(img)
    if erode:
        img = skimage.morphology.binary_erosion(img)


    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor((slide.dimensions[0]*1./patch_size*res/maxres-size_x)*patch_size*maxres/res)
    offset_y = np.floor((slide.dimensions[1]*1./patch_size*res/maxres-size_y)*patch_size*maxres/res)
    add_x = np.linspace(0,offset_x,size_x).astype(int)
    add_y = np.linspace(0,offset_y,size_y).astype(int)

    #list of sample pixels
    w = np.where(img>0)

    #grid=zip(w[1]*patch_size,w[0]*patch_size)
    grid = list(zip((w[1]*patch_size*maxres/res+add_x[w[1]]).astype(int),(w[0]*patch_size*maxres/res+add_y[w[0]]).astype(int)))

    #connectivity
    if overlap > 1:
        o = (add(overlap)*patch_size*maxres/res).astype(int)
        ox,oy = np.meshgrid(o,o)
        connx = np.zeros(img.shape).astype(bool)
        conny = np.zeros(img.shape).astype(bool)
        connd = np.zeros(img.shape).astype(bool)
        connu = np.zeros(img.shape).astype(bool)
        connx[:,:-1] = img[:,1:]
        conny[:-1,:] = img[1:,:]
        connd[:-1,:-1] = img[1:,1:]
        connu[1:,:-1] = img[:-1,1:] & ( ~img[1:,1:] | ~img[:-1,:-1] )
        connx = connx[w]
        conny = conny[w]
        connd = connd[w]
        connu = connu[w]
        extra = []
        for i,(x,y) in enumerate(grid):
            if connx[i]: extra.extend(zip(o+x,np.repeat(y,overlap-1)))
            if conny[i]: extra.extend(zip(np.repeat(x,overlap-1),o+y))
            if connd[i]: extra.extend(zip(ox.flatten()+x,oy.flatten()+y))
            if connu[i]: extra.extend(zip(x+ox.flatten(),y-oy.flatten()))
        grid.extend(extra)

    #prune squares
    if prune:
        level, mult = find_level(slide,res,maxres)
        psize = int(patch_size*mult)
        truegrid = []
        for tup in grid:
            reg = slide.read_region(tup,level,(psize,psize))
            # Enable OpenSlide caching
            reg = slide.read_region(tup,level,(psize,psize))
            if mult != 1:
                reg = reg.resize((224,224),Image.BILINEAR)
            reg = image2array(reg)
            if is_sample(reg,th/255,0.2,0.4,0.5):
                truegrid.append(tup)
    else:
        truegrid = grid

    return truegrid,img

def plot_extraction(slide,patch_size=224,res=20,min_cc_size=10,max_ratio_size=10,erode=False,prune=False,overlap=1,save=''):
    '''Script that shows the result of applying the detector in case you get weird results'''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if save:
        plt.switch_backend('agg')

    maxres = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    grid,_ = make_sample_grid(slide,patch_size,res,min_cc_size,max_ratio_size,erode,prune,overlap)
    thumb = slide.get_thumbnail((np.round(slide.dimensions[0]/50.),np.round(slide.dimensions[1]/50.)))


    ps = []
    for tup in grid:
        ps.append(patches.Rectangle(
            (tup[0]/50., tup[1]/50.), patch_size/50.*maxres/res, patch_size/50.*maxres/res, fill=False,
            edgecolor="red"
        ))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(thumb)
    for p in ps:
        ax.add_patch(p)
    if save:
        plt.savefig(save)
    else:
        plt.show()

#kernel=np.ones((5,5),np.uint8)
#cimg=cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

#out=[]
#for tup in grid:
#    out.append(sum([1 if x==tup else 0 for x in grid]))
#np.unique(np.array(out))
