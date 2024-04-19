
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# echo "Script directory: $SCRIPT_DIR"
PRJDIR=$SCRIPT_DIR/../../
CUDA_VISIBLE_DEVICES=0,1,2 python $PRJDIR/VL_py/main.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/intent_30" \
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
	--save_dir="$PRJDIR/data/intent_30" \
	--name='intent_30'
	# --test=1


