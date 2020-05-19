#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import cv2
import sys

INFO = {
    "description": "Robotpush Dataset",
    "url": "https://github.com/aeitel/robotpush_dataset",
    "version": "0.0.1",
    "year": 2018,
    "contributor": "Andreas Eitel",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'background',
        'supercategory': 'background',
    },
    {
        'id': 1,
        'name': 'object',
        'supercategory': 'object',
    },
]


def filter_for_png(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    #print("basename ",basename_no_extension,os.path.basename(image_filename))
    file_name_prefix = basename_no_extension + '_.*'
    #print("basename ",file_name_prefix)
    files = [os.path.join(root, f) for f in files]
    #print("Files0",files)
    files = [f for f in files if re.match(file_types, f)]
    #print("Files1",files)
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    #print("File2",files)
    return files

def main():


    ANNOTATION = "binary_masks"
    ROOT_DIR=sys.argv[1]
    IMAGE_DIR = os.path.join(ROOT_DIR, "train2014")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, ANNOTATION)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for png images
    for root, _, files in os.walk(IMAGE_DIR):
        files.sort()
        image_files = filter_for_png(root, files)
        image_files.sort()
        # go through each image
        for image_filename in image_files:
            print("Image file",image_filename)
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                files.sort()
                annotation_files = filter_for_annotations(root, files, image_filename)
                #print("Len annotation_files",len(annotation_files))
                #if (len(annotation_files) > 1):
                    #print("T",image_filename,annotation_files)
                    #exit(1)
                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = 1# np.random.randint(5)
                    print("class id",class_id)

                    category_info = {'id': class_id, 'is_crowd': 0}
                    #binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)
                    binary_mask = cv2.imread(annotation_filename,-1)
                    #img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    #ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
                    #binary_mask = cv2.bitwise_not(mask)
                    #cv2.imshow('maskBGR',binary_mask)
                    #cv2.waitKey(0)
                    #exit(1)
                    print("Segmentation id",segmentation_id)
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    if not os.path.exists(ROOT_DIR+'/annotations'):
        os.mkdir(ROOT_DIR+'/annotations')
    outname = ROOT_DIR + '/annotations' + '/instances_train2014.json'
    with open(outname, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("Saved json file",outname)

if __name__ == "__main__":
    main()
