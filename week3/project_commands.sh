python week3/create_labeled_queries.py
wc /workspace/datasets/fasttext/labeled_queries.txt
head /workspace/datasets/fasttext/labeled_queries.txt
cut -d' ' -f1 /workspace/datasets/fasttext/labeled_queries.txt | sort | uniq | wc

python week3/create_labeled_queries.py --min_queries 1000
shuf /workspace/datasets/fasttext/labeled_queries.txt > /workspace/datasets/fasttext/shuffled_labeled_queries.txt

head -50000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries_training_data.txt
tail -10000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries_test_data.txt

~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/queries_training_data.txt -output query_classifier -lr 0.5 -epoch 25 -wordNgrams 2
~/fastText-0.9.2/fasttext test query_classifier.bin /workspace/datasets/fasttext/queries_test_data.txt

