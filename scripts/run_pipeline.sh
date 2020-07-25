python -m disinformation_detection.train --random_seed 42 --model random_forest --features moral_feat style_feat complexity_feat bias_feat affect_feat
python -m disinformation_detection.train --random_seed 42 --model random_forest --features ngram
python -m disinformation_detection.train --random_seed 42 --model bert --batch_size 1 --lr 0.01 --epochs 2 --use_gpu true
