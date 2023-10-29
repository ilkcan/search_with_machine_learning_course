```bash
python createContentTrainingData.py --output /workspace/datasets/fasttext/labeled_products.txt

shuf /workspace/datasets/fasttext/labeled_products.txt --random-source=<(seq 99999) > /workspace/datasets/fasttext/shuffled_labeled_products.txt

head -n 10000 /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/training_data.txt
tail -n 10000 /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/test_data.txt

~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/training_data.txt -output /workspace/datasets/fasttext/product_classifier -lr 1.0 -epoch 25 -wordNgrams 2

~/fastText-0.9.2/fasttext predict /workspace/datasets/fasttext/product_classifier.bin -

~/fastText-0.9.2/fasttext test /workspace/datasets/fasttext/product_classifier.bin /workspace/datasets/fasttext/test_data.txt
~/fastText-0.9.2/fasttext test /workspace/datasets/fasttext/product_classifier.bin /workspace/datasets/fasttext/test_data.txt 5

cat /workspace/datasets/fasttext/shuffled_labeled_products.txt |sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_labeled_products.txt

python createContentTrainingData.py --output /workspace/datasets/fasttext/products.txt --label name
cat /workspace/datasets/fasttext/products.txt |  cut -c 10- | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_products.txt

~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/normalized_products.txt -output /workspace/datasets/fasttext/title_model
~/fastText-0.9.2/fasttext nn /workspace/datasets/fasttext/title_model.bin

~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/normalized_products.txt -output /workspace/datasets/fasttext/title_model_100 -minCount 100
~/fastText-0.9.2/fasttext nn /workspace/datasets/fasttext/title_model_100.bin
```


