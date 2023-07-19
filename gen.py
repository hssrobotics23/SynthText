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

import glob
import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
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
OUT_FILE = 'results/SynthText.h5'


def contour_area(contours):

    # create an empty list
    cnt_area = []
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))
 
    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area

def to_bounding_box(contours, n_boxes=1):

    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)
 
    # Loop through each contour of our image
    for i in range(0,len(contours),1):
        cnt = contours[i]
 
        # Only draw the the largest number of boxes
        maximum = cnt_area[min(n_boxes, len(cnt_area))-1]
        if (cv2.contourArea(cnt) >= maximum):
             
            # Use OpenCV boundingRect function to get the details of the contour
            x,y,w,h = cv2.boundingRect(cnt)
            yield (x, y, x + w, y + h)

def get_data(folder):
    keys = glob.glob(osp.join(folder, '*mask.png'))
    out = {
        "seg": {},
        "image": {},
        "area": {},
        "label": {}
    }
    for key in keys:
        im = Image.open(key)
        im_blur = Image.open(key)
        im_blur.filter(ImageFilter.BoxBlur(8))
        arr = np.asarray(im)
        arr_blur = np.asarray(im_blur)

        # Simple image color
        out["image"][key] = arr[:,:,:3]

        # Image blur and threshhold
        thresh_in = (arr_blur[:,:,3] < 127).astype(np.uint8)
        contours = cv2.findContours(thresh_in, 1, 2)[0]

        mask = np.zeros(thresh_in.shape, dtype=np.uint16)
      
        for (x0, y0, x1, y1) in to_bounding_box(contours, 1):
            mask[x0:x1, y0:y1] = 1

        out["seg"][key] = mask

        within = np.count_nonzero(mask)
        out["area"][key] = np.uint32([
            mask.size - within, within
        ])
        out["label"][key] = np.uint16([0, 1])
    return out


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    L = res[i]['txt']
    L = [n.encode("ascii", "ignore") for n in L]
    db['data'][dname].attrs['txt'] = L


def main(folder):
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  db = get_data(folder)
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print (colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
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
      noise = np.random.normal(ones, 0.1*ones)
      depth = ones + noise 
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print (colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=False)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('folder')
  args = parser.parse_args()
  main(args.folder)
