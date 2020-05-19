# Co-teaching,
MODEL='robotpush2316,remove_bg,1,flowfilter,15,actions,1,maxim,0,revels,maxepoch_10,ft_deepmask,lr_0,hfreq_1.0,coteaching,0.1,trial'
# Self-paced
#MODEL='robotpush2316,remove_bg,1,flowfilter,15,actions,1,maxim,0,revels,maxepoch_10,ft_deepmask,lr_0,hfreq_1.0,self-paced,0.1,trial'

TRAINING_TRIALS=5
RUN_EVAL=true
NMS_TRESH=0.4

# Evaluate multiple training runs
if [ "$RUN_EVAL" = true ]; then
  for tr in $(seq -w 1 $(printf "%01d" $TRAINING_TRIALS));
  do

    DATADIR='data/test_data'
    #Check images in folder
    NUM_IMAGES=$(ls -lh $DATADIR/1*/rgb_image/cloud*.png | wc -l)
    echo "Number of eval images: "$NUM_IMAGES

    #Evaluate all pushes
    MODELDIR='trained_models'
    # Note that we ignore detections outside the table workspace
    th evalPerImage.lua $MODELDIR/$MODEL$tr -datadir $DATADIR -split push-1 -crop_workspace -nms_overlap $NMS_TRESH
  done
fi

python pycocoTestDemo.py $MODEL $TRAINING_TRIALS $NMS_TRESH | tee $MODELDIR/$MODEL"1_nms"$NMS_TRESH/testlog
