#IMDIR='/home/eitel/code/singulation_segm/self_deepmask/data/test_data/6objects_aggregated_network'
IMDIR='/home/eitel/code/singulation_segm/self_deepmask/data/video/1509019242.185500975'
#MODELDIR='pretrained/deepmask'
MODELDIR='SelfDeepMask'
mkdir -p $IMDIR/paper
mkdir -p $IMDIR/paper/$MODELDIR
th computeProposals.lua paper/$MODELDIR -imdir $IMDIR/rgb_frames -area_max 0.01

#for imdir in $IMDIR/*; do
#    th computeProposals.lua $MODELDIR -imdir $imdir/rgb_image -area_max 0.01
#done
