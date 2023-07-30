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
import json
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
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 90 #max time per image in seconds

# path to the fonts etc
DATA_PATH = 'data'
JPEG_QUALITY = 75
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_DIR = 'results/images'

# Divide 51 sources into 17 classes
class_filter = {
    'sea salt': 'salt',
    'pickling salt': 'salt',
    'kosher salt': 'salt',

    'turmeric': 'tumeric',
    'mustard powder': 'tumeric',
    'mango powder': 'tumeric',

    'red pepper flakes': 'chiles',
    'chili powder': 'chiles',
    'sichuan pepper': 'chiles',

    'sumac': 'sumac',
    'smoked paprika': 'sumac',
    'paprika': 'sumac',

    'cinnamon': 'cinnamon',
    'chinese five-spice powder': 'cinnamon',
    'pumpkin pie spice': 'cinnamon',

    'mace': 'nutmeg',
    'garam masala': 'nutmeg',
    'nutmeg': 'nutmeg',

    'cumin': 'cumin',
    'curry powder': 'cumin',
    'ginger': 'cumin',

    'ground cloves': 'cloves',
    'dukkah': 'cloves',
    'poppy seeds': 'cloves',

    'cardamom': 'cardamom',
    'flax seeds': 'cardamom',
    'chia seeds': 'cardamom',

    'caraway seeds': 'caraway',
    'mahlab': 'caraway',
    'za atar seasoning': 'caraway',

    'coriander': 'coriander',
    'sesame seeds': 'coriander',
    'marjoram': 'coriander',

    'fennel seeds': 'fennel',
    'oregano': 'fennel',
    'celery seeds': 'fennel',

    'fenugreek': 'fenugreek',
    'annatto seeds': 'fenugreek',
    'carom seeds': 'fenugreek',

    'black pepper': 'peppercorns',
    'grains of paradise': 'peppercorns',
    'white pepper': 'peppercorns',

    'saffron': 'saffron',
    'ancho powder': 'saffron',
    'gochugaru': 'saffron',

    'allspice': 'allspice',
    'baharat seasoning': 'allspice',
    'garlic powder': 'allspice',

    'star anise': 'anise',
    'pickling spice': 'anise',
    'cayenne pepper': 'anise',
}
name_map = {

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
        k = found.groups(1)[0] if found else f_name

        # Only allow valid classes
        key = class_filter.get(k, None)
        if key == None:
            continue

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

  meta_data = []
  meta_file = os.path.join(OUT_DIR, 'index.json')
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  db = get_data(folder)
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  # create output directory
  Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
  print (colorize(Color.GREEN,'Storing the output in: '+OUT_DIR, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())

  RV3 = RendererV3(name_map, DATA_PATH, max_time=SECS_PER_IMG)
  for (i, imname) in enumerate(imnames):
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

      print(colorize(Color.RED,'%d of %d'%(i+1,len(imnames)), bold=True))
      res = RV3.render_text(imname,img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=False)
      if len(res) > 0:
        for (ri, r) in enumerate(res):
            word_verts = []
            lines = r["txt"]
            x_points, y_points = r["wordBB"]
            # TODO: fix this odd wordBB format!!
            for i in range(len(lines)):
                x_val = lambda j: int(round(x_points[j][i]))
                y_val = lambda j: int(round(y_points[j][i]))
                word_verts.append([
                    { "x": x_val(j), "y": y_val(j) } for j in range(4)
                ])
            # Each word has a four-point boundary
            words = [
                {"points": v, "word": w} for (v, w) in zip(word_verts, lines)
            ]
            out_name = f'{imname}-text-{ri}.jpg'
            fname = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(fname, r["img"][:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            print(f'Saved {out_name}')
            meta_data.append({
                'words': words,
                'file_name': out_name
            })
            # non-empty : successful in placing text:
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue

  # Write meta
  with open(meta_file, 'w') as wf:
      json.dump(meta_data, wf)


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('folder')
  args = parser.parse_args()
  main(args.folder)
