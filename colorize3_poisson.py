import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate as si
import scipy.ndimage as scim 
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
#import cPickle as cp
import _pickle as cp
#import Image
from PIL import Image
from poisson_reconstruct import blit_images
import pickle

def sample_weighted(p_dict):
    ps = p_dict.keys()
    return ps[np.random.choice(len(ps),p=p_dict.values())]

class Layer(object):

    def __init__(self,alpha,color):

        # alpha for the whole image:
        assert alpha.ndim==2
        self.alpha = alpha
        [n,m] = alpha.shape[:2]

        color=np.atleast_1d(np.array(color)).astype('uint8')
        # color for the image:
        if color.ndim==1: # constant color for whole layer
            ncol = color.size
            if ncol == 1 : #grayscale layer
                self.color = color * np.ones((n,m,3),'uint8')
            if ncol == 3 : 
                self.color = np.ones((n,m,3),'uint8') * color[None,None,:]
        elif color.ndim==2: # grayscale image
            self.color = np.repeat(color[:,:,None],repeats=3,axis=2).copy().astype('uint8')
        elif color.ndim==3: #rgb image
            self.color = color.copy().astype('uint8')
        else:
            print (color.shape)
            raise Exception("color datatype not understood")

class FontColor(object):

    def __init__(self, col_file):

        self.colorsRGB = np.uint8([
            # black and white
            [0, 0, 0],
            [255, 255, 255],
            # dark and light red
            [27,  0, 11],
            [21, 12, 16],
            [253,232,241],
            [255,165,203],
            # dark and light brown
            [31, 16,  0],
            [25, 19, 14],
            [255,245,234],
            [255,211,165],
            # dark and light blue
            [ 1, 11, 20],
            [ 9, 13, 16],
            [230,241,250],
            [171,217,255],
            # dark and light green
            [18, 29,  0],
            [19, 23, 13],
            [246,254,233],
            [221,255,165]
        ])
        self.ncol = self.colorsRGB.shape[0]
        colorsRGB = self.colorsRGB[None,:,:]
        self.colorsLAB = np.squeeze(cv.cvtColor(colorsRGB,cv.COLOR_RGB2Lab))


    def sample_normal(self, col_mean, col_std):
        """
        sample from a normal distribution centered around COL_MEAN 
        with standard deviation = COL_STD.
        """
        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype('uint8')

    def sample_from_data(self, bg_mat):
        """
        bg_mat : this is a nxmx3 RGB image.
        
        returns a tuple : (RGB_foreground, RGB_background)
        each of these is a 3-vector.
        """

        bg_orig = bg_mat.copy()
        bg_mat = cv.cvtColor(bg_mat, cv.COLOR_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]),3))
        bg_mean = np.mean(bg_mat,axis=0)

        norms = np.linalg.norm(self.colorsLAB-bg_mean[None,:], axis=1)
        # Closest background color
        bg_nn = np.argmin(norms)
        bg_col = self.colorsRGB[bg_nn] / 255
        bg_lightness = self.colorsLAB[bg_nn][0]

        # Optimal contrast ratio
        if bg_lightness < 127:
            return ((255, 255, 255), bg_col)
        return ((0, 0, 0), bg_col)

    def mean_color(self, arr):
        col = cv.cvtColor(arr, cv.COLOR_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]),3))
        col = np.mean(col,axis=0).astype('uint8')
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.COLOR_HSV2RGB))

    def invert(self, rgb):
        rgb = 127 + rgb
        return rgb

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(cv.cvtColor(rgb_color[None,None,:], cv.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128 #uint8 mods to 255
        col_comp = np.squeeze(cv.cvtColor(col_hsv[None,None,:],cv.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv.cvtColor(col1[None,None,:], cv.COLOR_RGB2HSV))
        col2 = np.squeeze(cv.cvtColor(col2[None,None,:], cv.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1 : h1,h2 = h2,h1 #swap
        dh = h2-h1
        if dh < 127: dh = 255-dh
        col1[0] = h1 + dh/2
        return np.squeeze(cv.cvtColor(col1[None,None,:],cv.COLOR_HSV2RGB))

    def change_value(self, col_rgb, v_std=50):
        col = np.squeeze(cv.cvtColor(col_rgb[None,None,:], cv.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0,1)
        ps = np.abs(vs - x/255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
        col[2] = 255*v_rand
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.COLOR_HSV2RGB))


class Colorize(object):

    def __init__(self, model_dir='data'):#, im_path):
        # # get a list of background-images:
        # imlist = [osp.join(im_path,f) for f in os.listdir(im_path)]
        # self.bg_list = [p for p in imlist if osp.isfile(p)]

        self.font_color = FontColor(col_file=osp.join(model_dir,'models/colors_new.cp'))

        # probabilities of different text-effects:
        self.p_bevel = 0.05 # add bevel effect to text
        self.p_outline = 0.05 # just keep the outline of the text
        self.p_drop_shadow = 0.25
        self.p_border = 0.0


    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        """
        alpha : alpha layer whose shadow need to be cast
        theta : [0,2pi] -- the shadow direction
        shift : shift in pixels of the shadow
        size  : size of the GaussianBlur filter
        op    : opacity of the shadow (multiplying factor)

        @return : alpha of the shadow layer
                  (it is assumed that the color is black/white)
        """
        if size%2==0:
            size -= 1
            size = max(1,size)
        shadow = cv.GaussianBlur(alpha,(size,size),0)
        [dx,dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op*sii.shift(shadow, shift=[dx,dy],mode='constant',cval=0)
        return shadow.astype('uint8')

    def border(self, alpha, size, kernel_type='RECT'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {'RECT':cv.MORPH_RECT, 'ELLIPSE':cv.MORPH_ELLIPSE,
                 'CROSS':cv.MORPH_CROSS}
        kernel = cv.getStructuringElement(kdict[kernel_type],(size,size))
        border = cv.dilate(alpha,kernel,iterations=1) # - alpha
        return border

    def merge_two(self,fore,back):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha/255.0
        a_b = back.alpha/255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f*a_b
        c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                + a_f[:,:,None]*c_f    )

        return Layer((255*a_r).astype('uint8'), c_r.astype('uint8'))

    def merge_down(self, layers):
        """
        layers  : [l1,l2,...ln] : a list of LAYER objects.
                 l1 is on the top, ln is the bottom-most layer.
        Note    : (1) it assumes that all the layers are of the SAME SIZE.
        @return : a single LAYER type object representing the merged-down image
        """
        nlayers = len(layers)
        if nlayers > 1:
            [n,m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2,-nlayers-1,-1):
                out_layer = self.merge_two(fore=layers[i], back=out_layer)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))
        
    def occlude(self):
        """
        somehow add occlusion to text.
        """
        pass

    def color_border(self, fg_lightness):
        """
        Decide on a color for the border:
        """
        return np.uint8((255,255,255) if fg_lightness < 127 else (0,0,0))

    def color_text(self, text_arr, h, bg_arr):
        """
        Decide on a color for the text:
            - could be some other random image.
            - could be a color based on the background.
                this color is sampled from a dictionary built
                from text-word images' colors. The VALUE channel
                is randomized.

            H : minimum height of a character
        """
        bg_col,fg_col,i = 0,0,0
        fg_col,bg_col = self.font_color.sample_from_data(bg_arr)
        return Layer(alpha=text_arr, color=fg_col), fg_col, bg_col


    def process(self, text_arr, bg_arr, min_h):
        """
        text_arr : one alpha mask : nxm, uint8
        bg_arr   : background image: nxmx3, uint8
        min_h    : height of the smallest character (px)

        return text_arr blit onto bg_arr.
        """
        # decide on a color for the text:
        l_text, fg_col, bg_col = self.color_text(text_arr, min_h, bg_arr)
        fg_lightness = cv.cvtColor(np.uint8([fg_col])[None], cv.COLOR_RGB2Lab)[0,0,0]
        bg_col = np.mean(np.mean(bg_arr,axis=0),axis=0)
        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'),color=bg_col)

        # Always use border if background varies
        bg_gray = cv.cvtColor(bg_arr, cv.COLOR_BGR2GRAY)
        small = tuple(max(1, d // 8) for d in bg_gray.shape)
        bg_gray_small = cv.resize(bg_gray, small, cv.INTER_LANCZOS4)
        bg_range = np.max(bg_gray_small) - np.min(bg_gray_small)
        # Shadow and border affected by contrast
        p_drop_shadow = max(0.25, 1.0 - (bg_range/255))
        p_border = min(1, 0.25 + (bg_range/255))

        l_text.alpha = l_text.alpha
        layers = [l_text]

        # add border:
        if np.random.rand() < p_border:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 2
            else: bsz = 4
            border_a = self.border(l_text.alpha, size=bsz)
            l_border = Layer(border_a, self.color_border(fg_lightness))
            layers.append(l_border)

        # add shadow:
        if np.random.rand() < p_drop_shadow:
            # shadow gaussian size:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 2
            else: bsz = 4

            # shadow angle:
            theta = np.pi/4 * np.random.choice([1,3,5,7]) + 0.5*np.random.randn()

            # shadow shift:
            if min_h <= 15 : shift = 1
            elif 15 < min_h < 30: shift = 5
            else: shift = 7

            # opacity:
            op = 0.40 + 0.1*np.random.randn()

            shadow = self.drop_shadow(l_text.alpha, theta, shift, 3*bsz, op)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
        

        layers.append(Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_col))
        l_normal = self.merge_down(layers)
        l_bg = Layer(alpha=255*np.ones_like(text_arr,'uint8'), color=bg_arr)

        if fg_lightness < 127:
            l_normal_out = cv.convertScaleAbs(l_normal.color, alpha=1.5, beta=0)
            l_out =  blit_images(l_normal_out,l_bg.color.copy())
        else:
            l_normal_out = l_normal.color
            l_out =  blit_images(l_normal_out,l_bg.color.copy())
        
        # plt.subplot(1,3,1)
        # plt.imshow(l_normal.color)
        # plt.subplot(1,3,2)
        # plt.imshow(l_bg.color)
        # plt.subplot(1,3,3)
        # plt.imshow(l_out)
        # plt.show()
        
        if l_out is None:
            # poisson recontruction produced
            # imperceptible text. In this case,
            # just do a normal blend:
            layers[-1] = l_bg
            return self.merge_down(layers).color

        return l_out


    def check_perceptible(self, txt_mask, bg, txt_bg):
        """
        --- DEPRECATED; USE GRADIENT CHECKING IN POISSON-RECONSTRUCT INSTEAD ---

        checks if the text after merging with background
        is still visible.
        txt_mask (hxw) : binary image of text -- 255 where text is present
                                                   0 elsewhere
        bg (hxwx3) : original background image WITHOUT any text.
        txt_bg (hxwx3) : image with text.
        """
        bgo,txto = bg.copy(), txt_bg.copy()
        txt_mask = txt_mask.astype('bool')
        bg = cv.cvtColor(bg.copy(), cv.COLOR_RGB2Lab)
        txt_bg = cv.cvtColor(txt_bg.copy(), cv.COLOR_RGB2Lab)
        bg_px = bg[txt_mask,:]
        txt_px = txt_bg[txt_mask,:]
        bg_px[:,0] *= 100.0/255.0 #rescale - L channel
        txt_px[:,0] *= 100.0/255.0

        diff = np.linalg.norm(bg_px-txt_px,ord=None,axis=1)
        diff = np.percentile(diff,[10,30,50,70,90])
        print ("color diff percentile :", diff)
        return diff, (bgo,txto)

    def color(self, bg_arr, text_arr, hs, place_order=None, x_pad=30, y_pad=5):
        """
        Return colorized text image.

        text_arr : list of (n x m) numpy text alpha mask (unit8).
        hs : list of minimum heights (scalar) of characters in each text-array. 
        text_loc : [row,column] : location of text in the canvas.
        canvas_sz : size of canvas image.
        
        return : nxmx3 rgb colorized text-image.
        """
        bg_arr = bg_arr.copy()
        if bg_arr.ndim == 2 or bg_arr.shape[2]==1: # grayscale image:
            bg_arr = np.repeat(bg_arr[:,:,None], 3, 2)

        # get the canvas size:
        canvas_sz = np.array(bg_arr.shape[:2])
        max_h, max_w = canvas_sz

        # initialize the placement order:
        if place_order is None:
            place_order = np.array(range(len(text_arr)))

        rendered = []
        for i in place_order[::-1]:
            # get the "location" of the text in the image:
            ## this is the minimum x and y coordinates of text:
            loc = np.where(text_arr[i])
            lx, ly = np.min(loc[0]), np.min(loc[1])
            mx, my = np.max(loc[0]), np.max(loc[1])
            l = np.array([lx,ly])
            m = np.array([mx,my])-l+1
            text_patch = text_arr[i][l[0]:l[0]+m[0],l[1]:l[1]+m[1]]

            # figure out padding:
            ext = canvas_sz - (l+m)
            num_pad = np.ones(4,dtype='int32')
            num_pad[:2] = np.minimum(num_pad[:2] * y_pad, l)
            num_pad[2:] = np.minimum(num_pad[2:] * x_pad, ext)
            text_patch = np.pad(text_patch, pad_width=((num_pad[0],num_pad[2]), (num_pad[1],num_pad[3])), mode='constant')
            l -= num_pad[:2]

            h,w = text_patch.shape
            bg = bg_arr[l[0]:l[0]+h,l[1]:l[1]+w,:]

            rdr0 = self.process(text_patch, bg, hs[i])
            rendered.append(rdr0)

            bg_arr[l[0]:l[0]+h,l[1]:l[1]+w,:] = rdr0#rendered[-1]


            return bg_arr

        return bg_arr
