SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PRJDIR=$SCRIPT_DIR/../../
# echo "Script directory: $SCRIPT_DIR"
python $SCRIPT_DIR/../data_process_VL.py \
    --source_path $SCRIPT_DIR/../../data/ \
    --target_path $SCRIPT_DIR/../../data/ \
    --switch_with_intent false \
    --use_augmented false \
#cp test data set to eval
cp $SCRIPT_DIR/../../data/test.csv $SCRIPT_DIR/../../data/eval.csv  
##vanilla
CUDA_VISIBLE_DEVICES=0,1,2 python $PRJDIR/VL_py/main.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/original" \
	--dataset= \
	--mc=0 \
	--epochs=30 \
	--lr=0.00003 \
	--qmax_words=395 \
	--amax_words=5 \
	--max_feats=64 \
	--batch_size=256 \
	--batch_size_val=256 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--feature_dim=512 \
	--dropout=0.3 \
	--seed=42 \
	--freq_display=2 \
	--save_dir="$PRJDIR/data/original" \
	--name='original'
	# --test=1

####
### intent
###
# echo "Script directory: $SCRIPT_DIR"
python $SCRIPT_DIR/../data_process_VL.py \
    --source_path $SCRIPT_DIR/../../data/ \
    --target_path $SCRIPT_DIR/../../data/ \
    --switch_with_intent true \
    --use_augmented false \
#cp test data set to eval
cp $SCRIPT_DIR/../../data/test.csv $SCRIPT_DIR/../../data/eval.csv  
##vanilla
CUDA_VISIBLE_DEVICES=0,1,2 python $PRJDIR/VL_py/main.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/intent" \
	--dataset= \
	--mc=0 \
	--epochs=30 \
	--lr=0.00003 \
	--qmax_words=395 \
	--amax_words=5 \
	--max_feats=64 \
	--batch_size=256 \
	--batch_size_val=256 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--feature_dim=512 \
	--dropout=0.3 \
	--seed=42 \
	--freq_display=2 \
	--save_dir="$PRJDIR/data/intent" \
	--name='intent'
	# --test=1



##
##augmented
##


####
### aug
###
# echo "Script directory: $SCRIPT_DIR"
python $SCRIPT_DIR/../data_process_VL.py \
    --source_path $SCRIPT_DIR/../../data/ \
    --target_path $SCRIPT_DIR/../../data/ \
    --switch_with_intent false \
    --use_augmented true \
#cp test data set to eval
cp $SCRIPT_DIR/../../data/test.csv $SCRIPT_DIR/../../data/eval.csv  
##vanilla
CUDA_VISIBLE_DEVICES=0,1,2 python $PRJDIR/VL_py/main.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/aug" \
	--dataset= \
	--mc=0 \
	--epochs=30 \
	--lr=0.00003 \
	--qmax_words=395 \
	--amax_words=5 \
	--max_feats=64 \
	--batch_size=256 \
	--batch_size_val=256 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--feature_dim=512 \
	--dropout=0.3 \
	--seed=42 \
	--freq_display=2 \
	--save_dir="$PRJDIR/data/aug" \
	--name='aug'
	# --test=1



####
### aug+intent
###
# echo "Script directory: $SCRIPT_DIR"
python $SCRIPT_DIR/../data_process_VL.py \
    --source_path $SCRIPT_DIR/../../data/ \
    --target_path $SCRIPT_DIR/../../data/ \
    --switch_with_intent true \
    --use_augmented true \
#cp test data set to eval
cp $SCRIPT_DIR/../../data/test.csv $SCRIPT_DIR/../../data/eval.csv  
##vanilla
CUDA_VISIBLE_DEVICES=0,1,2 python $PRJDIR/VL_py/main.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/aug_intent" \
	--dataset= \
	--mc=0 \
	--epochs=30 \
	--lr=0.00003 \
	--qmax_words=395 \
	--amax_words=5 \
	--max_feats=64 \
	--batch_size=256 \
	--batch_size_val=256 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--feature_dim=512 \
	--dropout=0.3 \
	--seed=42 \
	--freq_display=2 \
	--save_dir="$PRJDIR/data/aug_intent" \
	--name='aug_intent'
	# --test=1