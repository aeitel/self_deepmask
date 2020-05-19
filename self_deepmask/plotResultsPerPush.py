
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import os
import glob
plt.rcParams['figure.figsize'] = (10.0, 8.0)
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
print('Running demo for *%s* results.'%(annType))
#initialize COCO ground truth api

baseDir='/home/eitel/code/singulation_segm/self_deepmask/data/test_data'

# 05.2019
#modelDir = 'robotpush1329,remove_bg,1,flowfilter,15,oracle,100,revels,maxepoch_10,ft_deepmask,lr_0,hfreq_1.0,trial5_nms'
modelDir = args.model_dir
expDir = '6,8objects_aggregated_network'
dataTypes=['push00','push01','push02','push03','push04','push05','push06','push07','push08','push09','push10','push11']

dataDir = baseDir + '/' + expDir
#Evaluate all pushes
dataType = 'push-1'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)

trainingTrials = 5

baselines = ['deepmask1_nms0.5','sharpmask1_nms0.5']
RESULTS_BASELINES = np.zeros((len(baselines)+1,len(dataTypes)),np.float)
for b,baseline in enumerate(baselines):
    print("Baseline",baseline)
    for idx, dataType in enumerate(dataTypes):
        annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
        cocoGt=COCO(annFile)
        # Ours
        resDir = '/home/eitel/code/singulation_segm/self_deepmask/finalsubmission/'+ baseline + '/'+expDir
        resFile = resDir + '/' + dataType +'/jsons'
        print("Resfile",resFile)
        resFile =  glob.glob(resFile+"/*.json")[0]
        #print("res",resFile)
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
            average_precision = cocoEval.stats[1]
            RESULTS_BASELINES[b,idx] = average_precision
            #anns = cocoDt.dataset['annotations']
            #for id, ann in enumerate(anns):
            #   print("ID",id,ann)
        else:
            print("File does not exist: ",resFile)
            exit(1)


RESULTS_PER_PUSH = np.zeros((trainingTrials,len(dataTypes)),float)

for trial in range(trainingTrials):
    for idx, dataType in enumerate(dataTypes):
        annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
        cocoGt=COCO(annFile)
        # Ours
        modelDirTrial = modelDir+str(trial+1)+'_nms'+str(args.nms_treshold)
        resDir = '/home/eitel/code/singulation_segm/self_deepmask/arxiv/'+ modelDirTrial + '/'+expDir
        print("Resdir",resDir)
        if trial == 0:
            resDir1 = resDir
        resFile = resDir + '/' + dataType +'/jsons'
        resFile =  glob.glob(resFile+"/*.json")[0]
        #print("res",resFile)
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
            average_precision = cocoEval.stats[1]
            RESULTS_PER_PUSH[trial, idx] = average_precision
            #average_precisions[b,idx] = average_precision
            #anns = cocoDt.dataset['annotations']
            #for id, ann in enumerate(anns):
            #   print("ID",id,ann)
        else:
            print("File does not exist: ",resFile)
            exit(1)

OUR_MEAN_RESULTS_PER_PUSH = np.mean(RESULTS_PER_PUSH, axis=0)
OUR_STDDEV_RESULTS_PER_PUSH = np.std(RESULTS_PER_PUSH, axis=0)

print("MEAN RESULTS PER PUSH",OUR_MEAN_RESULTS_PER_PUSH)
print("STDDEV RESULTS PER PUSH",OUR_STDDEV_RESULTS_PER_PUSH)

fontsz = 40
markersz = 16
linew = 2
plt.rc('font',**{'family':'cmu-serif','sans-serif':['Helvetica']})
plt.rc('text',usetex=True)
x_axis= np.linspace(0,len(dataTypes)-1, num=len(dataTypes))
plt.clf()
plt.xlabel('number of pushes',fontsize=fontsz)
plt.ylabel('AP@0.5',fontsize=fontsz)
#plt.ylim([20,80])

plt.gcf().set_size_inches(14, 9, forward=True)
plt.tick_params(axis='both', which='major', labelsize=fontsz)
#plt.hlines(average_precisions[0], 0, len(average_precisions)-1, colors='k', linestyles='--')

plt.plot(x_axis, RESULTS_BASELINES[0], label='DeepMask with NMS', marker='o', markersize=markersz, linestyle='-',linewidth=linew,color='orange',zorder=-32)
plt.plot(x_axis, RESULTS_BASELINES[1], label='SharpMask with NMS', marker='o', markersize=markersz, linestyle='-',linewidth=linew,color='red',zorder=-32)
#plt.plot(x_axis, OUR_MEAN_RESULTS_PER_PUSH,label='ours',marker='o', markersize=markersz, linestyle='-',linewidth=linew,color='blue')
plt.errorbar(x_axis, OUR_MEAN_RESULTS_PER_PUSH, label='SelfDeepMask', capsize=2,yerr=OUR_STDDEV_RESULTS_PER_PUSH,markersize=markersz,linewidth=linew,color='blue',fmt='-o');


plt.legend(loc="upper left", ncol=1, shadow=False, fontsize=fontsz-6)
outfile = resDir1 + '/deepmask_ap.pdf'
print("Saving",outfile)
plt.savefig(outfile)
plt.show()
