TRAINING_RUNS=5
for tr in $(seq -w 1 $(printf "%01d" $TRAINING_RUNS));
do
echo "Training run "$tr

#SelfDeepMask, 2.3k interactions
EXPDIR="robotpush2316,remove_bg,1,flowfilter,15,actions,1,maxim,0"

LEARNINGRATE=0
HFREQ=1.0
GPU=0
MACHINE='revels'
MAXEPOCH=10
FINETUNE='deepmask'
DATADIR="data/train_data/"$EXPDIR
TRAINDIR=$DATADIR"/train2014"
TRAINSIZE=$(ls -lh $TRAINDIR/*.png | wc -l)
VALDIR=$DATADIR"/val2014"
VALSIZE=$(ls -lh $VALDIR/*.png | wc -l)
BATCHSIZE=32
NOISEFILTER='coteaching'    # nofilter,self-paced,reedhard,coteaching
COTEACHFORGETRATE=0.1
echo "Data: "$DATADIR
echo "Train: "$TRAINDIR
echo "Validation: "$VALDIR
echo "Training Set: "$TRAINSIZE
echo "Validation Set: "$VALSIZE

# Run deepmask
export CUDA_VISIBLE_DEVICES=$GPU
MODEL=$EXPDIR','$MACHINE',maxepoch_'$MAXEPOCH',ft_'$FINETUNE,'lr_'$LEARNINGRATE,'hfreq_'$HFREQ,$NOISEFILTER,$COTEACHFORGETRATE,'trial'$tr
th train.lua -rundir 'trained_models'/$MODEL -datadir $DATADIR -maxload 100 -testmaxload 49 -finetune pretrained/$FINETUNE -maxepoch $MAXEPOCH -lr $LEARNINGRATE -hfreq $HFREQ -noisefilter $NOISEFILTER -coteachforgetrate $COTEACHFORGETRATE
done
