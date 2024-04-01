## Environment

- Python 3.8.0
- PyTorch 2.0.0
- CUDA 11.7
- Transformers 4.27.4

Prepare the anaconda environment:

```bash
conda create -n t2t python=3.8.0
conda activate t2t
pip install -r requirements.txt
```

## Run the experiments

### 1. Prepare the data

Please download the datasets mentioned in the HW instruction:
- [data](https://drive.google.com/drive/folders/1RAWWGTI7ciFkQfl3P9TSlC8Wm-seZYrN)


First, `mkdir data`, put the downloaded data into the `./data` folder. The file structure should be like this:

``` 
-- data
    |-- train
    |   |-- train.csv
    |-- test
    |   |-- test.csv
    |-- videos
    |-- Other-dialogue-transcripts
```

Then, run the following command to preprocess the data:

```bash
python data_process.py
```

It will process the origin `train.csv` and `test.csv` into a new csv format (pls refer to `data_process.py` for more details) that can be used for text-to-indices classification, where the labels are all converted to indices (`0` or `1`).   

The data statistics are as follows:

```
train samples: 5100
test samples: 1290
```

Add `--split` can randomly split 10% samples from training set as validation set to tune the hyperparameters, however, since we only have 5100 samples, currently I don't think it's necessary to split the data.


### 2. Train the model


For text-to-indices classification, run the following command:

```bash
sh scripts/run_sen_cls.sh [GPU] [batch_size] [model_name] [learning_rate]

# for example
sh scripts/run_sen_cls.sh 7 32 t5-small 5e-4   ### T5-Small
sh scripts/run_sen_cls.sh 7 32 t5-base 5e-4    ### T5-Base
sh scripts/run_sen_cls.sh 7 16 t5-large 5e-4   ### T5-Large
sh scripts/run_sen_cls.sh 7 4 t5-3b 1e-4       ### T5-XL
```

Please adjust the hyperparameters according to your needs, such as the enocder model, learning rate, batch size, etc.

I can run the above commands on the single-GPU server (with only 20K MiB). You can adjust the batch size to fit your GPU memory.


### 3. Results

The results are saved in `./out/cls` .

**Renze: I have tried with different size T5 encoder. You guys can feel free to use any other encoder model.**

The current performances by me are as follows:

| Encoder Model    | ACC      | F1       |
|----------|----------|----------|
| T5-small | 84.031   | 81.2343  |
| T5-base  | 84.3411  | 81.1029  |
| T5-large | 82.5581  | 78.8023  |
| T5-3b    | **85.5039**  | **82.4379**  |



### 4. How to improve the performance?

Our current accuracy on the test set is already high (~85%). To further improve the performance, among the 5 hints mentioned in the HW instruction, I guess the following three directions are more effective (according to my experience):

- **Data augmentation**: according to the performances I got, it seems like the larger model can not achieve significantly better performance than the smaller model. So, I guess this is due to the limited training data. 
- **Additional data**: similarly, try to use the additional data under `Other-dialogue-transcripts`.
- **Leveraging the video data**: try to use the video data under `videos`.

