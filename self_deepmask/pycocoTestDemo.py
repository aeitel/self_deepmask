
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import os
import glob
#plt.rcParams['figure.figsize'] = (10.0, 8.0)
import argparse

parser = argparse.ArgumentParser(description='Description')
# Required positional argument
parser.add_argument('model_dir', type=str,
                    help='model directory')
parser.add_argument('training_trials',type=int,help='number of models to evaluate')
parser.add_argument('nms_treshold',type=float,help='nms treshold')

args = parser.parse_args()


annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
#print 'Running demo for *%s* results.'%(annType)
#initialize COCO ground truth api

baseDir='data/test_data'

modelDir = args.model_dir
#Evaluate all pushes
dataType = 'push-1'
annFile = '%s/annotations/%s_%s.json'%(baseDir,prefix,dataType)
cocoGt=COCO(annFile)

trainingTrials = 5
RESULTS = np.zeros((trainingTrials,12),float)
for trial in range(trainingTrials):

    modelDirTrial = modelDir+str(trial+1)+'_nms'+str(args.nms_treshold)
    resDir = 'trained_models/'+ modelDirTrial + '/test_data' 
    resFile = resDir + '/' + dataType +'/jsons'
    print("res",resFile)
    resFile =  glob.glob(resFile+"/*.json")[0]

    if os.path.isfile(resFile):
        print(resFile)
        cocoDt=cocoGt.loadRes(resFile)
        imgIds=sorted(cocoGt.getImgIds())
        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.catIds = [1]
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        RESULTS[trial] = cocoEval.stats
        #with open(resDir+"/coco_eval.txt", "w") as f:
        #    print >> f, cocoEval.stats
MEAN_RESULTS = np.mean(RESULTS, axis=0)
STDDEV_RESULTS = np.std(RESULTS,axis=0)
print(RESULTS)
print("MEAN",MEAN_RESULTS)
print("STDDEV",STDDEV_RESULTS)
