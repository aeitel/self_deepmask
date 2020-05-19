#EVAL_MAX_PUSH_NUMBER=8
#DATADIR='/home/eitel/code/singulation_segm/self_deepmask/data/test_data/6objects_aggregated_network'

EVAL_MAX_PUSH_NUMBER=11
#DATADIR='/home/eitel/code/singulation_segm/self_deepmask/data/test_data/8objects_aggregated_network'
#DATADIR='/home/eitel/code/singulation_segm/self_deepmask/data/test_data/6,8objects_aggregated_network'
DATADIR='/home/eitel/code/singulation_segm/self_deepmask/data/val_data/val2014'

NUM_MASKS=$(ls -lh $DATADIR/1*/binary_masks/*.png | wc -l)

for i in $(seq 0 $EVAL_MAX_PUSH_NUMBER);
do echo $i
python shapes_to_coco.py $DATADIR $i
done
# Create test set for all pushes
python shapes_to_coco.py $DATADIR -1

echo "Number of masks in "$DATADIR": "$NUM_MASKS
