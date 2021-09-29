import struct
import random
import numpy as np
from pdb import set_trace

class BMPWriter:
  def __init__(self):
    self.palettes = {'standard': [(255,255,255),(255,0,0),(0,0,255),(255,255,0),(128,128,128),(0,255,0),(255,128,0),(255,0,255),(0,255,255),(0,128,0),(128,0,0),(0,0,128),(128,128,0),(128,0,128),(255,128,128)]
                   }

  def makebmpheader(self,wd,ht,WD):
    # write bitmap header
    _bitmap = struct.pack("c", b'B')
    _bitmap += struct.pack("c", b'M')
    _bitmap += struct.pack("I", 118 + ht*(WD))#size in bytes of the file
    _bitmap += struct.pack("I", 0)
    _bitmap += struct.pack("I", 118 )#after how many bytes data starts
    _bitmap += struct.pack("I", 40)#DIB header size
    _bitmap += struct.pack("I", wd)
    _bitmap += struct.pack("I", ht)
    _bitmap += struct.pack("H", 1)#color planes
    _bitmap += struct.pack("H", 4)#bpp
    _bitmap += struct.pack("I", 0)#compression
    _bitmap += struct.pack("I", ht * (wd) )#size of image only
    _bitmap += struct.pack("I", 0)
    _bitmap += struct.pack("I", 0)
    _bitmap += struct.pack("I", 16)#number of colors
    _bitmap += struct.pack("I", 0)
    return _bitmap

  def makebmppalette(self,palette):
    '''
    palette must be a list of tuples (r,g,b)
    '''
    _bitmap = b''
    for r,g,b in palette:
      _bitmap += struct.pack("B", b) + struct.pack("B", g) + struct.pack("B", r) + struct.pack("B", 0)
    if len(palette)<16:
      for i in range(16-len(palette)):
        _bitmap += struct.pack("I", 0)
    return _bitmap

  def makebmp(self,data,w,h,palette):
    '''
    data should be a ndarray of dtype uint8
    palette is a list of tuples (r,g,b)
    '''
    h = self.makebmpheader(w,h,data.shape[1])
    p = self.makebmppalette(palette)
    return h + p + data[::-1,:].tobytes()

  def writebmp(self,fname,data,w,h,palette='standard'):
    p = self.palettes[palette]
    bmp = self.makebmp(data,w,h,p)
    f = open(fname, 'wb')
    f.write(bmp)
    f.close()

  def getrowsize(self,w):
    W = int(np.ceil(w/2.))
    stride = (4 - (W % 4)) % 4
    return W + stride

  def makeempty(self,w,h):
    W = self.getrowsize(w)
    return np.zeros((h,W),dtype='uint8')
