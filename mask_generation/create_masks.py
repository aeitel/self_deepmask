# Copyright (c) 2020,Andreas Eitel, All rights reserved.

from lib import flowlib as fl
from PIL import Image
import numpy as np
import os
import cv2
import glob
from skimage import data, segmentation, color
from skimage.future import graph
import random
import matplotlib.patches as mpatches
import argparse

CROPX = (80,720)
CROPY =(120,520)
random.seed(1)


def load_action(filename):
  if os.path.isfile(filename):
    action = np.loadtxt(filename, delimiter=' ', unpack=True)
  else:
    action = np.array([0,0,0],dtype=np.int_)
    #np.savetxt(filename,np.atleast_2d(action),fmt='%d')
    print("Warning: No action file found for",filename)

  angle_deg = action[2]
  if (angle_deg < 0.0):
    angle_deg = (180+angle_deg)+180
    #print("Deg",angle_deg)
  action[2] = angle_deg*np.pi/180

  return action

def normalized_cut(img):
  labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=0)
  out1 = color.label2rgb(labels1, img, kind='avg',bg_label=-1)

  g = graph.rag_mean_color(img, labels1, mode='similarity')
  labels2 = graph.cut_normalized(labels1, g)
  out2 = color.label2rgb(labels2, img, kind='avg',bg_label=-1)
  return out2

def remove_background2(img,mask):
  res = cv2.bitwise_and(img,img,mask = mask)
  # Remove white pixels
  #ret, mask2 = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 253, 0, cv2.THRESH_TOZERO_INV)
  #res2 = cv2.bitwise_and(res,res,mask = mask2)
  #res2[np.where(res2 == [0])] = [255]
  res[np.where(res < [5])] = [255]
  return res

def treshold_magn(flow,mag,tresh):
  res = np.copy(flow)
  res[mag < tresh] = [0,0]
  return res

def has_large_flow_magn(mag,tresh):
    #print("Flow magn",np.mean(mag))
    if (np.mean(mag) > tresh):
        print("Has large flow magn",np.mean(mag))
        return True
    else:
        return False

def has_large_flow_stddev(mag,img,tresh):
    if (np.std(mag[np.where(img == [255])]) > tresh):
        print("Has large flow std dev",np.std(mag[np.where(img == [255])]))
        return True
    else:
        return False

def treshold_ang(flow,ang,tresh,tresh_delta):
  # apply +-tresh around angle
  res = np.copy(flow)
  below_tresh = np.where(ang < tresh+tresh_delta)
  above_tresh = np.where(ang > tresh-tresh_delta)

  return res

def get_label(img,x,y):
    label = img[y,x]
    return label

def crop_image(image,x1,x2,y1,y2):
    cropped = np.zeros(image.shape,image.dtype)
    cropped = image[y1:y2,x1:x2]
    return cropped

def filter_out(img):
    is_out = False
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(mask)
    n_white_pix = np.sum(img == 255)
    #print("White pix",n_white_pix)
    if (n_white_pix > 0.025*img.shape[1]*img.shape[0]):
        is_out = True
    if (n_white_pix < 100):
        is_out = True
    return is_out

def sample_nearest_label(img,pixel):
    # Sample start til you find a segment
    sample_x = np.random.normal(pixel[0], 10, 50)
    sample_y = np.random.normal(pixel[1], 10, 50)
    label = get_label(img,int(pixel[0]),int(pixel[1]))
    for j in range(0,len(sample_x)):
        x = min(img.shape[1]-1,max(0,int(sample_x[j])))
        y = min(img.shape[0]-1,max(0,int(sample_y[j])))
        label = get_label(img,x,y)
        if ((label < 253).any()):
            break
    return label

def main():

    USE_CLUSTERING = True
    DEBUG_LEN = 0
    SAVE_ORIG=True
    CROP_IMAGE=False
    BINARY_MASK=True
    FLOWDIR='flow'
    IMAGEDIR = 'train2014'
    MASKDIR = 'binary_masks'
    USE_REVERSE_FLOW=False
    if USE_REVERSE_FLOW:
        FLOWDIR='flow_reverse'

    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("expdir", help="experiment dir where result is saved")
    parser.add_argument("-remove_bg", help="use wienseg depth data to remove background", default=1, required=False)
    parser.add_argument("-flowfilter", help="use flow to filter large angular std dev", default=0, required=False)
    parser.add_argument("-use_actions", help="use push actions for segment filtering ", default=1, required=False)
    parser.add_argument("-max_images", help=" ", default=0, required=False)


    args = parser.parse_args()

    INDIR = args.indir
    EXPDIR = args.expdir
    REMOVE_BACKGROUND = int(args.remove_bg)
    FLOWFILTER = int(args.flowfilter)
    USE_ACTIONS = int(args.use_actions)
    MAX_IMAGES = int(args.max_images)

    print('Unlabeled data directory',args.indir)
    print('Results directory',args.expdir)

    print('USE_CLUSTERING',USE_CLUSTERING)
    print('USE_ACTIONS',USE_ACTIONS)
    print('REMOVE_BACKGROUND',REMOVE_BACKGROUND)
    print('DEBUG_LEN',DEBUG_LEN)
    print('SAVE_ORIG',SAVE_ORIG)
    print("USE FLOW FILTER",FLOWFILTER)
    print("MAX_IMAGES",MAX_IMAGES)


    if not os.path.exists(args.expdir+'/'+IMAGEDIR):
        os.mkdir(args.expdir+'/'+IMAGEDIR)
    if not os.path.exists(args.expdir+'/'+MASKDIR):
        os.mkdir(args.expdir+'/'+MASKDIR)
    if not os.path.exists(args.expdir+'/'+'outliers'):
        os.mkdir(args.expdir+'/'+'outliers')

    count = 0
    filenames = sorted(glob.glob(INDIR+'/'+FLOWDIR+'/*.flo'))
    start_index = -1
    for index,filename in enumerate(filenames):
        if MAX_IMAGES:
            if count == MAX_IMAGES-1:
                exit(1)
        if index < start_index:
            continue
        #if USE_ACTIONS:
        if (USE_REVERSE_FLOW):
            action_file = filenames[index-1].replace(FLOWDIR,'actions')
        else:
            action_file = filenames[index].replace(FLOWDIR,'actions')

            action_file = action_file.replace('.flo','.txt')
            action = load_action(action_file)
        if not np.any(action):
            continue

        flow = fl.read_flow(filename)
        orig = fl.flow_to_image(flow)
        #orig = cv2.imread(filename.replace(FLOWDIR,'flow_image').replace('.flo','.png'),-1)
        rgb_im = cv2.imread(filename.replace(FLOWDIR,'rgb_image').replace('.flo','.png'),-1)

        if SAVE_ORIG:
            im_file = filename.replace(FLOWDIR,FLOWDIR+'_image').replace('.flo','.png')
            cv2.imwrite(im_file,orig)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if FLOWFILTER:
            if has_large_flow_magn(mag,7.0):
                continue

        im_file = filename.replace(FLOWDIR,MASKDIR)
        im_file = im_file.replace('.flo','.png')

        img = fl.flow_to_image(flow)
        #img = cv2.imread(filename.replace(FLOWDIR,'flow_image'),-1)

        infile = im_file.replace(MASKDIR,'wienseg')
        wienseg = cv2.imread(infile, 0)
        retwienseg, wiensegmask = cv2.threshold(wienseg, 10, 255, cv2.THRESH_BINARY)
        if REMOVE_BACKGROUND:
            img = remove_background2(img,wiensegmask)
            #cv2.imwrite(im_file.replace('.png','back.png'),img)

        if USE_CLUSTERING:
            img = normalized_cut(img)
            #cv2.imwrite(im_file.replace('.png','cluster.png'),img)

        if REMOVE_BACKGROUND:
            #infile = im_file.replace(MASKDIR,'wienseg')
            img = remove_background2(img,wiensegmask)

        if USE_ACTIONS:
            radius = 100
            start = (int(action[0]),int(action[1]))
            xend = min(img.shape[1]-1,max(0,int(action[0]+radius*np.cos(action[2]))))
            yend = min(img.shape[0]-1,max(0,int(action[1]-radius*np.sin(action[2]))))
            end = (xend,yend)
            label1 = get_label(img,start[0],start[1])
            label2 = get_label(img,end[0],end[1])
            if ((label1 >= 253).all()):
                label1 = sample_nearest_label(img,start)

            if (USE_REVERSE_FLOW):
                mask_label = cv2.inRange(img,label2,label2)
            else:
                mask_label = cv2.inRange(img,label1,label1)
            # Bitwise-AND mask and original image
            img = cv2.bitwise_and(img,img, mask= mask_label)
            img[np.where(img == [0])] = [255]
            #if DEBUG_LEN > 0:
                #cv2.circle(img,(int(action[0]),int(action[1])),radius,128,thickness=1)
                #cv2.line(img,start,end,(255,0,0),5)
                #cv2.imshow('maskBGR',img)
                #cv2.waitKey(0)
                #exit(1)

        if CROP_IMAGE:
            img = crop_image(img,CROPX[0],CROPX[1],CROPY[0],CROPY[1])
            rgb_im = crop_image(rgb_im,CROPX[0],CROPX[1],CROPY[0],CROPY[1])

        if filter_out(img):
           print("Skipped large mask",im_file)
           continue

        if BINARY_MASK:
            img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 253, 255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(mask)
            n_white_pix = np.sum(img == 255)
            if (n_white_pix < 100):
                #Skip small binary_masks
                continue

        if FLOWFILTER:
            if has_large_flow_stddev(mag,img,FLOWFILTER):
                print("Skipped",im_file)
                outlierfile = im_file.replace(MASKDIR,'outliers').replace(INDIR,EXPDIR)
                cv2.imwrite(outlierfile,img)
                continue

        if USE_REVERSE_FLOW:
            mask_file = mask_file.replace('.png','rev.png')

        mask_file = im_file.replace('.png','_0.png').replace(INDIR,EXPDIR)
        print("Save binary mask",mask_file)
        cv2.imwrite(mask_file,img)
        train_file = im_file.replace(MASKDIR,IMAGEDIR).replace(INDIR,EXPDIR)
        print("Save",train_file)
        cv2.imwrite(train_file,rgb_im)

        if DEBUG_LEN > 0:
            #cv2.imshow('orig',orig)
            #cv2.waitKey(0)
            #cv2.imshow('mod',img)
            #cv2.waitKey(0)
            if (count == DEBUG_LEN):
                exit(1)

        count+=1
        if count % 10 == 0:
            print("Number of masks saved in",args.indir+'/'+MASKDIR,count)

if __name__ == "__main__":
    main()
