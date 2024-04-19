This repository contains a basic pipeline for the binary classification task (mentioned in [CSE 582 final project instruction](https://psu.instructure.com/courses/2309886/assignments/15994501)), where I use a language model encoder (i.e., [T5](https://huggingface.co/docs/transformers/model_doc/t5)) with a simple linear classifier head.

The metrics are accuracy and F1 score.

>Note:
>This is just a basic implementation for this task. Further performance improvements can be made by using additional data, data augmentation, and leveraging the video data.

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


First, make a data folder `mkdir data`; then, put the downloaded data into the `./data` folder. The file structure should be like this:

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

I can run the above commands on the single-GPU server (with only 20 GB). You can adjust the batch size to fit your GPU memory.


### 3. Results

The results are saved in `./out/cls` . For example, you can find the performance of `t5-3b` in file `./out/cls/t5-3b/predict_results.json`.

**I have tried with different size T5 encoder. You guys can feel free to use any other encoder model.**

The current performances by me are as follows:

| Encoder Model    | ACC      | F1       |
|----------|----------|----------|
| T5-small | 84.03   | 81.23  |
| T5-base  | 84.34  | 81.10  |
| T5-large | 82.55  | 78.80  |
| T5-3b    | **85.50**  | **82.43**  |

Experimental Result:
| Encoder Model    | Original Dataset without intent | Original Dataset with intent|Augmented Dataset without intent|Augmented Dataset with intent|
|----------|----------|----------|----------|----------|
| T5-small | ACC:83.9    F1:81.3  Precision:80.9 Recall:81.6 | ACC:85.6    F1:83.0 Precision:83.1 Recall:82.8 |ACC:83.9    F1:81.1  Precision:80.9 Recall:81.2 | ACC:84.3    F1:81.2 Precision:81.8 Recall:80.7 |
| T5-base  | ACC:84.1    F1:80.9  Precision:81.6 Recall:80.2  | ACC:85.7    F1:82.7 Precision:83.6 Recall:81.9  |ACC:83.3    F1:80.1  Precision:80.4 Recall:79.5  | ACC:85.5    F1:82.6 Precision:83.3 Recall:81.9|
| T5-large | ACC:83.5    F1:80.2 Precision:80.8  Recall:79.5  | ACC:83.8    F1:80.1 Precision:81.7 Recall:78.9 |ACC:84.3    F1:81.3 Precision:81.7 Recall:81.0  | ACC:83.9    F1:80.1 Precision:82.2 Recall:78.6 |
| roberta-base | ACC:85.6    F1:76.3 Precision:  Recall:  | ACC:86.1    F1:77.1 Precision:  Recall: |ACC:85.4    F1:75  Precision:   Recall:81.0 | ACC:85.2    F1:74.4 Precision:  Recall: |
| distilroberta-base | ACC:84.3    F1:73.5  Precision:  Recall: | ACC:85.4    F1:75.3 Precision:  Recall: |ACC:84.9    F1:73.8  Precision:  Recall: | ACC:85.1    F1:74.5 Precision:  Recall: |





### 4. How to improve the performance?

**Our current accuracy on the test set is already high (~85%)**. To further improve the performance, among the 5 hints mentioned in the HW instruction, I guess the following three directions are more effective for performance improvement (according to my experience):

- **Data augmentation**: according to the performances I got, it seems like the larger model can not achieve significantly better performance than the smaller model. So, I guess this is due to the limited training data. 
- **Additional data**: similarly, try to use the additional data under `Other-dialogue-transcripts`.
- **Leveraging the video data**: try to use the video data under `videos`.

