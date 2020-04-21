
import glob, os, json, sys
from os import sep as S
from tqdm import tqdm

DIR = '/Users/Peter/Dropbox/datasets/text/aclImdb'


for set in ['train', 'test']:

    result = {}
    result['pos'], result['neg'] = [], []

    for sentiment in ['pos', 'neg']:
        print('loading', sentiment)

        for file in tqdm(glob.glob(f'{DIR}{S}{set}{S}{sentiment}{S}*.txt')):
            with open(file, 'r') as f:
                result[sentiment].append(f.read())

    with open(f'{DIR}{S}imdb.{set}.json', 'w') as out:
        json.dump(result, out)

print('done.')