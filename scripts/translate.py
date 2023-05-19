#!/usr/bin/env python

import os
import sys
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()


def deep_translator_translate(x):
    translated = GoogleTranslator(source='es', target='en').translate(x)
    return translated

def translate(filename):
    df = pd.read_csv(filename)
    df['original_text'] = df['text']
    df['text'] = df['original_text'].progress_apply(deep_translator_translate)
    return df

def main():
    path = sys.argv[1]
    translated = translate(path)

    parent = os.path.dirname(os.path.dirname(path))
    out_dir = os.path.join(parent, 'es2en/')

    basename = os.path.basename(path)
    out_filename = basename.replace('.csv', '_translated.csv')
    full_path = os.path.join(out_dir, out_filename)
    print(f'Saved: {full_path}')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    translated.to_csv(full_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
