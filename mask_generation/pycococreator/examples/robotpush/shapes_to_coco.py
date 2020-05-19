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
from os.path import relpath
import sys

#ROOT_DIR = '/home/eitel/code/singulation_segm/self_deepmask/data/test_data/6objects_aggregated_network'
ANNOTATION_DIR = "binary_masks"
IMAGE_DIR = "rgb_image"
#EVAL_PUSH_NUMBER = -1

INFO = {
    "description": "Robotpush Dataset",
    "url": "TODO",
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
    {
        'id': 2,
        'name': 'workspace',
        'supercategory': 'background',
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
    file_name_prefix = basename_no_extension + '_.*'
    files = [os.path.join(root, f) for f in files]
    #print("Files0",files)
    files = [f for f in files if re.match(file_types, f)]
    #print("Files1",files)
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    #print("File2",files)
    return files

def main():

    EVAL_PUSH_NUMBER = -1
    ROOT_DIR=sys.argv[1]
    if len(sys.argv) == 3:
        EVAL_PUSH_NUMBER=int(sys.argv[2])

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for trial_dir in os.listdir(ROOT_DIR):
        if os.path.isdir(os.path.join(ROOT_DIR, trial_dir)):
            print(trial_dir)
            image_dir = os.path.join(ROOT_DIR, trial_dir, IMAGE_DIR)
            annotation_dir = os.path.join(ROOT_DIR, trial_dir, ANNOTATION_DIR)
            print("image dir",image_dir)
            print("annotation dir",annotation_dir)
            # filter for png images
            for root, _, files in os.walk(image_dir):
                files.sort()
                if (EVAL_PUSH_NUMBER > len(files)-1):
                   #files = files[len(files)-1:len(files)]
                   continue
                elif (EVAL_PUSH_NUMBER >= 0):
                   files = files[EVAL_PUSH_NUMBER:EVAL_PUSH_NUMBER+1]
                image_files = filter_for_png(root, files)
                image_files.sort()
                #go through each image
                for image_filename in image_files:
                    #print("image filename",image_filename)
                    image = Image.open(image_filename)
                    image_info = pycococreatortools.create_image_info(
                    image_id, relpath(image_filename,ROOT_DIR), image.size)
                    coco_output["images"].append(image_info)

                    # filter for associated png annotations
                    for root, _, files in os.walk(annotation_dir):
                        files.sort()
                        annotation_files = filter_for_annotations(root, files, image_filename)
                        if (len(annotation_files) == 0):
                            print("No annotations for", image_filename)
                            coco_output["images"].remove(image_info)
                            break
                        for annotation_filename in annotation_files:
                            print(annotation_filename)
                            class_id = 1
                            # Assign workspace class
                            if annotation_filename.find("_11.png") > -1:
                                class_id = 2
                            #print("class id",class_id)

                            category_info = {'id': class_id, 'is_crowd': 0}
                            binary_mask = cv2.imread(annotation_filename,-1)
                            print("Segmentation id",segmentation_id)
                            annotation_info = pycococreatortools.create_annotation_info(
                                segmentation_id, image_id, category_info, binary_mask,
                                image.size, tolerance=2)

                            if annotation_info is not None:
                                coco_output["annotations"].append(annotation_info)

                            segmentation_id = segmentation_id + 1

                    image_id = image_id + 1

    basename = os.path.basename(ROOT_DIR)
    basename = ''.join([i for i in basename if not i.isdigit()])
    if not os.path.exists(ROOT_DIR + '/annotations'):
        os.makedirs(ROOT_DIR + '/annotations')
    outname = ROOT_DIR + '/annotations' + '/instances_push' + str(EVAL_PUSH_NUMBER).zfill(2) + '.json'
    with open(outname, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("Saved json file",outname)

if __name__ == "__main__":
    main()
