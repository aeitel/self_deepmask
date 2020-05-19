# Ignore push number for training
EXP='robotpush2316,remove_bg,1,flowfilter,15,actions,1,maxim,0'
DATADIR='../../../data/train/'
TRAINDIR='../../../../self_deepmask/data/train_data/'

python shapes_to_coco_train.py $DATADIR$EXP
mkdir -p $TRAINDIR$EXP
echo 'Copy training data '$DATADIR$EXP' to '$TRAINDIR$EXP
cp -r $DATADIR$EXP'/train2014' $TRAINDIR$EXP
cp -r $DATADIR$EXP'/val2014' $TRAINDIR$EXP
cp -r $DATADIR$EXP'/annotations' $TRAINDIR$EXP
