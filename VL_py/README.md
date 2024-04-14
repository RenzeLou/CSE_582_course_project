# Enviroment setup
pip install -r requirements.txt

## install clip
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

# preprocessing
1. download data using scripts/install.sh
2. extract video feature using VL_py/extract/extract_clip_features.ipynb
3. to preprocess the dataset run 
```bash
bash shells/data_preprocess.sh`
```

# running
```bash
bash shells/agqa_v2_mist.sh`
```

# Results
