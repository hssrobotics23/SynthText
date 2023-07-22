from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
import re
#import cPickle as cp
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math
from common import *
import pickle

def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps,p=ps)]

def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:,None,None]

def crop_safe(arr, rect, bbs=[], x_pad=30, y_pad=5):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= y_pad
    rect[2:] += x_pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0],v0[1]:v1[1],...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i,0] -= v0[0]
            bbs[i,1] -= v0[1]
        return arr, bbs
    else:
        return arr


class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, name_map, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {0.0 : 'WORD',
                       1.0 : 'LINE'}

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.95
        self.max_shrink_trials = 10 # 0.9^10 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_font_h = 25 #px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 75 #px

        # text-source : gets english text:
        self.text_source = TextSource(name_map)

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    def render_multiline(self,font,lines):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lengths = [len(l) for l in lines]
        space = font.get_rect('O')

        # font parameters:
        line_spacing = font.get_sized_height() + 1
        
        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0*line_bounds.width), round(1.25*line_spacing*len(lines)))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        x, y = 0, 0
        for l in lines:
            x = 0 # carriage-return
            y += line_spacing # line-feed

            # Quick kerning hack
            kernings = []
            ch_widths = []
            for ch in l:
                ch_bounds = font.render_to(surf, (0,0), ch)
                ch_widths.append(ch_bounds.width)

            for (i, ch) in enumerate(l):
                if i == 0 or l[i-1].isspace() or l[i].isspace():
                    kernings.append(space.width * font.char_spacing)
                    continue
                mean_width = (ch_widths[i] + ch_widths[i-1])/2
                kernings.append(mean_width * font.char_spacing)

            for ch in l: # render each character
                if ch.isspace(): # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x,y), ch)
                    ch_bounds.x = x + ch_bounds.x + kernings[i]
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width + kernings[i]
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = ' '.join(lines)

        # crop the surface to fit the text:
        bbs = np.array(bbs)

        #text_arrs[i][:32,:] = np.clip(120 - 0.5*cv2.convertScaleAbs(text_arrs[i][:32,:], 0.5), 80, 180)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs)
        surf_arr = surf_arr.swapaxes(0,1)
#        self.visualize_bb(surf_arr,bbs) #TODO
        return surf_arr, bbs 


    def get_nline_nchar(self,mask_size,font_height,font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H,W = mask_size
        nline = int(np.ceil(H/(2*font_height)))
        nchar = int(np.floor(W/font_width))
        return nline,nchar

    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:            
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            '''
            ba[ba > 127] = 1e5
            intersect = ssig.fftconvolve(ba,ta[::-1,::-1],mode='valid')
            safemask = intersect < 1e5

            if not np.any(safemask): # no collision-free position:
                #warn("COLLISION!!!")
                return back_arr,locs[:i],bbs[:i],order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]),:]
            '''
            ba_h, ba_w = ba.shape
            ta_h, ta_w = ta.shape

            # Simply center the text!
            cy, cx = (ba_h / 2, ba_w / 2)
            loc_list = [ cy - ta_h / 2, cx - ta_w / 2]
            loc = np.clip(np.int32(loc_list), a_min=0, a_max=None)

            bbs[i] = move_bb(bbs[i],loc[::-1])

            # blit the text onto the canvas
            h,w = text_arrs[i].shape
            out_arr[loc[0]:loc[0]+h,loc[1]:loc[1]+w] += text_arrs[i][:h,:w]


        return out_arr, locs, bbs, order

    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.max(np.sum(m,axis=0))
        rW = np.max(np.sum(m,axis=1))
        return rH,rW

    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords


    def render_sample(self,imname,font,mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        #H,W = mask.shape
        space = font.get_rect('O')
        H,W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)

        text_type = sample_weighted(self.p_text)
        text = self.text_source.sample(imname,text_type)
        lines = text.split('\n')
        max_n_char = max([len(t) for t in lines])

        scaling = 1.5
        max_font_w = W / max_n_char
        max_font_h = min(H, (1/f_asp)*max_font_w)
        max_font_h = scaling * min(max_font_h, self.max_font_h)

        # find the maximum height in pixels:
        if max_font_h < self.min_font_h: # not possible to place any text here
            return #None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            f_h_px = max_font_h * self.f_shrink
            f_h = self.font_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px 
            i += 1

            font.size = f_h # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline,nchar = self.get_nline_nchar(mask.shape[:2],f_h,f_h*f_asp)
            #print "  > nline = %d, nchar = %d"%(nline, nchar)

            # sample text:
            if len(lines)==0 or np.any([len(line)==0 for line in lines]):
                continue
            #print colorize(Color.GREEN, text)

            # render the text:
            txt_arr,bb = self.render_multiline(font, lines)
            bb = self.bb_xywh2coords(bb)

            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]):
                #warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask,loc,bb, _ = self.place_text([txt_arr], mask, [bb])
            if len(loc) > 0:#successful in placing the text collision-free:
                return text_mask,loc[0],bb[0],lines

        return #None


    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(ta, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=255, thickness=20)

        text_arr[:] = ta


class FontState(object):
    """
    Defines the random state of the font rendering  
    """
    size = [200, 10]  # normal dist mean, std
    strong = 0.5
    wide = 0.5
    strength = [0.03, 0.01]  # uniform dist in this interval
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun

    def __init__(self, data_dir='data'):

        char_freq_path = osp.join(data_dir, 'models/char_freq.cp')        
        font_model_path = osp.join(data_dir, 'models/font_px2pt.cp')

        # get character-frequencies in the English language:
        with open(char_freq_path,'rb') as f:
            #self.char_freq = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path,'rb') as f:
            #self.font_model = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.font_model = p
            
        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
        self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]


    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12 # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars,size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes,w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:,[3,4]]        
            r = np.abs(sizes[:,1]/sizes[:,0]) # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w*r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0]*font_size_px + m[1] #linear model


    def sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            'strong': np.random.rand() < self.strong,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': 0.2 + 0.05 * np.random.rand(),
            'capsmode': random.choice(self.capsmode)
        }

    def init_font(self,fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.char_spacing = fs['char_spacing']
        font.strong = fs['strong']
        font.strength = fs['strength']
        font.antialiased = True
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, sentences.
    """
    def __init__(self, name_map):
        """
        TXT_FN : path to file containing text data.
        """
        self.name_map = name_map
        self.fdict = {'WORD':self.sample_word,
                      'LINE':self.sample_line}

    def sample(self,imname,kind='WORD'):
        return self.fdict[kind](imname)
        
    def sample_word(self,imname):
        found = re.search(r"spice-(.+)", imname)
        key = found.groups(1)[0] if found else imname
        full_word = key.replace('-', ' ')
        word = self.name_map.get(full_word, full_word)
        print(f'{word}', '' if word == full_word else f'({full_word})')

        return word

    def sample_line(self,imname):
        line = self.sample_word(imname)
        return '\n'.join(line.split(' '))
