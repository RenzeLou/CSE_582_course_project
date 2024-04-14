
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# echo "Script directory: $SCRIPT_DIR"
python $SCRIPT_DIR/../data_process_VL.py --source_path $SCRIPT_DIR/../../data/ --target_path $SCRIPT_DIR/../../data/

#cp test data set to eval

cp $SCRIPT_DIR/../../data/test.csv $SCRIPT_DIR/../../data/eval.csv  