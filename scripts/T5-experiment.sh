#!/bin/sh

echo "data processing w/ no intent no augmented"
python data_process.py 


sh scripts/run_sen_cls.sh 1,2,3 32 t5-small 5e-4  t5-small-vanilla ### T5-Small
sh scripts/run_sen_cls.sh 1,2,3 32 t5-base 5e-4   t5-base-vanilla ### T5-Base
sh scripts/run_sen_cls.sh 1,2,3 16 t5-large 5e-4  t5-batch16-large-vanilla ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 4 t5-3b 1e-4      t5-3b-batch4-vanilla ### T5-XL
sh scripts/run_sen_cls.sh 1,2,3 32 t5-large 5e-4   t5-large-32-vanilla ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 32 t5-3b 5e-4       t5-3b-32-vanilla ### T5-XL

echo "data processing w/ intent"
python data_process.py --switch_with_intent

sh scripts/run_sen_cls.sh 1,2,3 32 t5-small 5e-4   t5-small-intent ### T5-Small
sh scripts/run_sen_cls.sh 1,2,3 32 t5-base 5e-4    t5-base-intent ### T5-Base
sh scripts/run_sen_cls.sh 1,2,3 16 t5-large 5e-4   t5-batch16-large-intent ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 4 t5-3b 1e-4       t5-3b-batch4-intent ### T5-XL
sh scripts/run_sen_cls.sh 1,2,3 32 t5-large 5e-4   t5-large-32-intent ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 32 t5-3b 5e-4       t5-3b-32-intent ### T5-XL

echo "data processing w/ augmented"
python data_process.py --use_augmented
sh scripts/run_sen_cls.sh 1,2,3 32 t5-small 5e-4   t5-small-aug ### T5-Small
sh scripts/run_sen_cls.sh 1,2,3 32 t5-base 5e-4    t5-base-aug ### T5-Base
sh scripts/run_sen_cls.sh 1,2,3 16 t5-large 5e-4   t5-batch16-large-aug ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 4 t5-3b 1e-4       t5-3b-batch4-aug ### T5-XL
sh scripts/run_sen_cls.sh 1,2,3 32 t5-large 5e-4   t5-large-32-aug ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 32 t5-3b 5e-4       t5-3b-32-aug ### T5-XL

echo "data processing w/ intent and augmented"
python data_process.py --switch_with_intent --use_augmented

sh scripts/run_sen_cls.sh 1,2,3 32 t5-small 5e-4   t5-small-aug_intent ### T5-Small
sh scripts/run_sen_cls.sh 1,2,3 32 t5-base 5e-4    t5-base-aug_intent ### T5-Base
sh scripts/run_sen_cls.sh 1,2,3 16 t5-large 5e-4   t5-large-batch16-aug_intent ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 4 t5-3b 1e-4       t5-3b-batch4-aug_intent ### T5-XL
sh scripts/run_sen_cls.sh 1,2,3 32 t5-large 5e-4   t5-large-aug_intent ### T5-Large
sh scripts/run_sen_cls.sh 1,2,3 32 t5-3b 5e-4       t5-3b-aug_intent ### T5-XL
