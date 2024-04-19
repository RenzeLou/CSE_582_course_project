This is an adaption of the original mist model (https://github.com/showlab/mist/tree/main)
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
3. to preprocess the dataset run (to set different data, use --switch_with_intent and   --use_augmented )
```bash
bash shells/data_preprocess.sh`
```
# running single training
```bash
bash shells/VL_classfication.sh`
```

# running all experiments on mist
```bash
bash shells/experiment.sh`
```

# Model Hyperparameter
The learning rate is set to 3e-5
Hyperparameters:
```
Epochs: 20 
Learning rate (lr): 0.00003
Maximum words in question (qmax_words): 395
Maximum words in answer (amax_words): 5
Maximum features (max_feats): 64
Batch size for training: 256
Batch size for validation: 256
Number of thread readers: 8
Masked Language Model probability (mlm_prob): 0
Number of layers in the model (n_layers): 2
Embedding dimension (embd_dim): 512
Feed-forward dimension (ff_dim): 1024
Feature dimension: 512
Dropout rate: 0.3
Random seed: 42
```
# Results
For the results please view:
Overall, Adding intent label helps to improve the accuracy by around 1%. (83 to 84-85)
https://wandb.ai/jkys/CSE_582_course_project-VL_py/reports/Mist---Vmlldzo3NTM3NjA5
