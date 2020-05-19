DATASET='robotpush2316'
INDIR='data/train/'
REMOVE_BACKGROUND=1
# Some value between 10 and 20 is good, allowed angle std dev in flow field, 0 means no flow filtering
FLOWFILTER=15
USE_ACTIONS=1
MAX_IMAGES=0

EXPERIMENT=$DATASET',remove_bg,'$REMOVE_BACKGROUND,'flowfilter',$FLOWFILTER,'actions',$USE_ACTIONS,'maxim',$MAX_IMAGES
mkdir -p $INDIR$EXPERIMENT
mkdir -p $INDIR$EXPERIMENT'/annotations'
echo $INDIR$EXPERIMENT
rsync -azv $INDIR$DATASET'/val2014' $INDIR$EXPERIMENT
rsync -azv $INDIR$DATASET'/annotations/instances_val2014.json' $INDIR$EXPERIMENT'/annotations'
python create_masks.py $INDIR$DATASET $INDIR$EXPERIMENT -remove_bg  $REMOVE_BACKGROUND -flowfilter $FLOWFILTER -use_actions $USE_ACTIONS -max_images $MAX_IMAGES

echo "Number of images: "$(ls $INDIR$EXPERIMENT'/'train2014/*.png | wc -l)
echo "Number of binary masks: "$(ls $INDIR$EXPERIMENT'/'binary_masks/*.png | wc -l)
