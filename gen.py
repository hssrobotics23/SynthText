# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import re
import glob
import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from pathlib import Path
from PIL import ImageFilter
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 90 #max time per image in seconds

# path to the fonts etc
DATA_PATH = 'data'
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_DIR = 'results/images'

# Names
name_map = {
    'ancho powder': 'ancho',
    'annatto seeds': 'annatto',
    'baharat seasoning': 'baharat',
    'black pepper': 'pepper',
    'carom seeds': 'carom',
    'cayenne pepper': 'cayenne',
    'chinese five spice powder': 'Chinese five spice',
    'chipotle powder': 'chipotle',
    'cream of tartar': 'tartar',
    'curry powder': 'curry',
    'fennel seeds': 'fennel',
    'garlic powder': 'garlic',
    'ground cloves': 'cloves',
    'mojo seasoning': 'mojo',
    'mustard-powder': 'mustard',
    'old bay seasoning': 'Old Bay',
    'pumpkin pie spice': 'pumpkin spice',
    'sea salt': 'sea salt',
    'za atar seasoning': 'za\'atar'
}


def max_contour(contours):

    # create an empty list
    cnt_area = []
    cnt_out = []
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))
        cnt_out.append(contours[i])
 
    max_index = cnt_area.index(max(cnt_area))
    return cnt_out[max_index]

def to_bounding_box(contours):

    # Call our function to get the max contour area
    cnt = max_contour(contours)
     
    # Use OpenCV boundingRect function to get the details of the contour
    x,y,w,h = cv2.boundingRect(cnt)
    return (x, y, x + w, y + h)

def get_data(folder):
    paths = glob.glob(osp.join(folder, '*mask.png'))
    out = {
        "seg": {},
        "image": {},
        "area": {},
        "label": {}
    }
    for path in paths:
        im = Image.open(path)
        arr = np.asarray(im).copy()
        arr_blur = cv2.GaussianBlur(arr, (15, 15), 15)

        # Key for name lookup
        f_name = Path(path).name
        found = re.search(r"(prompt-.+)-mask", f_name)
        key = found.groups(1)[0] if found else f_name

        # Image threshhold
        thresh_mask = (arr[:,:,3] < 127) 
        thresh_in = thresh_mask.astype(np.uint8)
        contours = cv2.findContours(thresh_in, 1, 2)[0]
        if (len(contours) == 0):
            print(f'Skipping {f_name}')
            continue

        # Select only largest contour
        mask = np.zeros(thresh_in.shape, dtype=bool)
        (x0, y0, x1, y1) = to_bounding_box(contours)
        mask[y0:y1, x0:x1] = thresh_mask[y0:y1, x0:x1]

        # blur image under mask
        (nz_y,nz_x) = thresh_in.nonzero()
        arr[nz_y,nz_x,:] = arr_blur[nz_y,nz_x,:]

        out["seg"][key] = np.uint16(mask)
        out["image"][key] = arr[:,:,:3]
        within = np.count_nonzero(mask)
        out["area"][key] = np.uint32([within])
        out["label"][key] = np.uint16([1])

    keys = ''.join([f'"{k}"' for k in out["area"].keys()])
    print(f'loaded {keys}')
    return out

def to_radial(shape):
   Y = np.linspace(-1, 1, shape[0])[None, :]
   X = np.linspace(-1, 1, shape[1])[:, None]
   return np.sqrt(X**2 + Y**2)

def main(folder):
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  db = get_data(folder)
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  # create output directory
  Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
  print (colorize(Color.GREEN,'Storing the output in: '+OUT_DIR, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(name_map, DATA_PATH, max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img_array = db['image'][imname]
      img = Image.fromarray(img_array[:])
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['area'][imname]
      label = db['label'][imname]

      # re-size uniformly:
      sz = img_array.shape[:2][::-1]
      ones = np.ones(sz[::-1], dtype=np.float32)
      depth = ones - .125 * to_radial(sz) 
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print (colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      res = RV3.render_text(imname,img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=False)
      if len(res) > 0:
        for (ri, r) in enumerate(res):
            out_name = f'{imname}-text-{ri}.png'
            fname = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(fname, r["img"][:,:,::-1])
            print(f'Saved {out_name}')
            # non-empty : successful in placing text:
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('folder')
  args = parser.parse_args()
  main(args.folder)
