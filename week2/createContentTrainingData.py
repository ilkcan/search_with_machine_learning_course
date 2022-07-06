import argparse
import multiprocessing
import glob
from tqdm import tqdm
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import warnings

# Ignore warning for append
warnings.simplefilter(action='ignore', category=FutureWarning)

stemmer = SnowballStemmer("english")

def transform_name(product_name):
    # Transform names to lowercase
    ret = product_name.lower()
    # Remove non-alphanumeric characters other than space, hyphen, or period.
    ret = ''.join(c for c in ret if c.isalpha() or c.isnumeric() or c=='-' or c==' ' or c =='.')
    # Apply Snowball stemmer
    # ret = ' '.join(map(stemmer.stem, ret.split(' ')))
    return ret

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

# Location for category data
categoriesFilename = '/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--categories_file", default=categoriesFilename, help="The location of the categories file")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# Setting min_product_names removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category.")

# Setting cat_depth reduces the granularity of classification to improve precision.
general.add_argument("--cat_depth", default=0, type=int, help="The maximum depth in the tree for categories (0 means no maximum)")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min and below the max
min_products = args.min_products

sample_rate = args.sample_rate
cat_depth = args.cat_depth
categoriesFilename = args.categories_file

catMap = {}
tree = ET.parse(categoriesFilename)
root = tree.getroot()
for child in root:
    catPath = child.find('path')
    leafCat = catPath[len(catPath) - 1].find('id').text
    if cat_depth > 0:
        truncDepth = min(cat_depth, len(catPath) - 1)
        truncCat = catPath[truncDepth].find('id').text
        catMap[leafCat] = truncCat
    else:
        catMap[leafCat] = leafCat

df = pd.DataFrame(columns = ['Category', 'Name'])

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Skip categores that are not in the category file.
                      if cat not in catMap:
                          continue
                      # Map to maximum category depth if specified.
                      cat = catMap[cat]
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      df = df.append({'Category' : cat, 'Name' : transform_name(name)}, ignore_index = True)


# If min_products is specified, filter out labels with too few products.
if min_products != 0:
    catGroups = df.groupby('Category').size().reset_index(name='Size')
    catGroups = catGroups[catGroups['Size'] >= min_products]
    cats = set(catGroups.Category)
    df = df[df['Category'].isin(cats)]
#Shuffle
df = df.sample(frac=1)
# Print output.
with open(output_file, 'w') as output:
    print("Writing results to %s" % output_file)
    for index, row in df.iterrows():
        output.write("__label__%s %s\n" % (row['Category'], row['Name']))