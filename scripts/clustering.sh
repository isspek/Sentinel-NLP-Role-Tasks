python -m clustering.clustering --model doc2vec_ner --random_seed 42 --inputs text
python -m clustering.clustering --model ner --random_seed 42 --inputs text
python -m clustering.clustering --model doc2vec --random_seed 42 --inputs text

python -m clustering.clustering --model doc2vec_ner --random_seed 42 --inputs title
python -m clustering.clustering --model ner --random_seed 42 --inputs title
python -m clustering.clustering --model doc2vec --random_seed 42 --inputs title