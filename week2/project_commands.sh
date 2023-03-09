## Create labeled data
python week2/createContentTrainingData.py --output /workspace/datasets/fasttext/labeled_products.txt

## Checks
head -1 /workspace/datasets/fasttext/labeled_products.txt
# __label__abcat0107029 Recoton - 1/8" Mini Stereo 3.5mm Y Adapter

tail -1 /workspace/datasets/fasttext/labeled_products.txt
# __label__abcat0101001 LG 55" Class LED 1080p Smart 3D HDTV, Blu-ray Player & 3D Glasses Package

## Shuffle data
shuf /workspace/datasets/fasttext/labeled_products.txt --random-source=<(seq 99999) > /workspace/datasets/fasttext/shuffled_labeled_products.txt

## Training/Test Split
head -10000 /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/training_data.txt
tail -10000 /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/test_data.txt

wc -l /workspace/datasets/fasttext/training_data.txt
wc -l /workspace/datasets/fasttext/test_data.txt

## Training
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/training_data.txt -output product_classifier -lr 1.0 -epoch 25 -wordNgrams 2

## Test single instance
~/fastText-0.9.2/fasttext predict product_classifier.bin -

## Test using test data
~/fastText-0.9.2/fasttext test product_classifier.bin /workspace/datasets/fasttext/test_data.txt

## Normalize data
cat /workspace/datasets/fasttext/shuffled_labeled_products.txt |sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_labeled_products.txt

## Create pruned labeled data
python week2/createContentTrainingData.py --min_products 50 --output /workspace/datasets/fasttext/pruned_labeled_products.txt

## Level2 - Synonyms Data
python week2/createContentTrainingData.py --output /workspace/datasets/fasttext/products.txt --label name

## Normalize products data
cat /workspace/datasets/fasttext/products.txt |  cut -c 10- | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_products.txt

## Train model
~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/normalized_products.txt -output title_model_100  -epoch 25 -minCount 100

## Test model
~/fastText-0.9.2/fasttext nn title_model_100.bin

## Top words
cat /workspace/datasets/fasttext/normalized_products.txt | tr " " "\n" | grep "...." | sort | uniq -c | sort -nr | head -1000 | grep -oE '[^ ]+$' > /workspace/datasets/fasttext/top_words.txt

## Create synonyms file
python week2/createSynonyms.py > /workspace/datasets/fasttext/synonyms.csv

## Docker cp
docker cp /workspace/datasets/fasttext/synonyms.csv opensearch-node1:/usr/share/opensearch/config/synonyms.csv

./index-data.sh -r -p /workspace/search_with_machine_learning_course/week2/conf/bbuy_products.json