import pandas as pd
import fasttext

model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")

with open('/workspace/datasets/fasttext/top_words.txt') as top_words_file:
    words = top_words_file.read().splitlines()

result = []

for word in words:
    synonyms = []
    synonyms.append(word)
    nns = model.get_nearest_neighbors(word)
    for nn in nns:
        if nn[0] > 0.75:
            synonyms.append(nn[1])
    result.append(",".join(synonyms))

print(result)

with open('/workspace/datasets/fasttext/synonyms.csv', mode='wt') as output_file:
    output_file.write('\n'.join(result))