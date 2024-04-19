#pip install gdown
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# echo "Script directory: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
mkdir -p tmp data
cd tmp
gdown --folder https://drive.google.com/drive/folders/1EI5GCXGWq9eL-O8PhIxOP9GFyHHh3i6k

unzip CSE*/*.zip -d ../data
cd $SCRIPT_DIR/../
mv data/*project/* data/
rm data/*project tmp -r