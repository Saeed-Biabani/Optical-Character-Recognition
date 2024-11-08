from hazm import word_tokenize, Normalizer
from src.dataset import ImageDataset
from src.vocab import Vocab
import pprint
import pickle
import tqdm
import os

root = 'Dataset/Persian-OCR-230k/data/'

normalizer = Normalizer()
tokenizer = lambda x : word_tokenize(normalizer.normalize(x))

df = ImageDataset.load_data(root = root, split = 'train')

tokens = [
    tokenizer(item) for item in tqdm.tqdm(
        df['text'].to_list(),
        colour = 'magenta',
    )
]
max_len = len(max(tokens, key = len))
print(max_len)

specials = ['[PAD]', '[UNK]', '[SOS]', '[EOS]']

vocab = Vocab()
vocab.initVocab(
    tokens,
    min_freq = 2,
    specials = specials 
);
vocab_path = os.path.join(root, "vocab.pkl")
vocab.saveVocab(vocab_path)
print(f"Num Tokens : {len(vocab)}")

info_path = os.path.join(root, 'info.pkl')
with open(info_path, 'wb') as f:
    data = {
        'pad_idx' : specials.index('[PAD]'),
        'sos_idx' : specials.index('[SOS]'),
        'eos_idx' : specials.index('[EOS]'),
        'vocab_path' : vocab_path,
        'max_positions' : max_len + 8
    }
    pickle.dump(data, f)

with open(os.path.join(root, 'info.pkl'), 'rb') as f:
    data = pickle.load(f)
    pprint.pprint(data, sort_dicts = False)