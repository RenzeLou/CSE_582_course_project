
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# echo "Script directory: $SCRIPT_DIR"
PRJDIR=$SCRIPT_DIR/../../
CUDA_VISIBLE_DEVICES=6 python $PRJDIR/VL_py/main_agqa_v2.py --dataset_dir="$PRJDIR/data/" \
	--feature_dir="$PRJDIR/data/"  \
	--checkpoint_dir="$PRJDIR/data/aug_intent" \
	--dataset= \
	--mc=0 \
	--epochs=20 \
	--lr=0.00003 \
	--qmax_words=395 \
	--amax_words=5 \
	--max_feats=32 \
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
	# --test=1


