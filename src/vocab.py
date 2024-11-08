from collections import Counter, OrderedDict
from typing import Iterable, Optional
import pickle

class Vocab:
    def __init__(self) -> None:
        self.vocab = None
        self.stoi = None
        self.itos = None
        self.default_token = None
        self.specials = None

    def initVocab(
        self,
        iterator : Iterable,
        min_freq : int = 1,
        specials : Optional[tuple[str]] = None
    ) -> None:
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        tokens = []
        for token, freq in ordered_dict.items():
            if freq >= min_freq:
                tokens.append(token)

        if specials:
            tokens[0:0] = specials

        self.vocab = tokens
        self.specials = specials
        self.default_token = '[UNK]'

        self.initStoi(tokens)
        self.initItos(tokens)

    def initStoi(self, tokens : tuple[str]) -> None:
        self.stoi = {item:indx for indx, item in enumerate(tokens)}

    def initItos(self, tokens : tuple[str]) -> None:
        self.itos = {indx:item for indx, item in enumerate(tokens)}

    def __len__(self) -> None:
        return len(self.vocab)

    def __getitem__(self, token : str) -> int:
        return self.stoi.get(
            token,
            self.stoi[self.default_token]
        )

    def __contains__(self, token : str) -> bool:
        return token in self.vocab

    def encode(
        self,
        tokens : Iterable[str]
    ) -> Iterable[int]:
        enc = []
        for token in tokens:
            enc.append(self[token])
        return enc

    def decode(
        self,
        ids : Iterable[int],
        ignore_special  : bool = False
    ) -> Iterable[str]:
      res = []
      for item in ids:
        n = self.lookup_token(item)
        if ignore_special:
          if not n in self.specials:
            res.append(n)
        else:
          res.append(n)
      return res

    def lookup_token(self, token : int) -> str:
        return self.itos[token]

    def saveVocab(self, fname : str = 'vocab.pkl') -> None:
        with open(fname, 'wb') as f:
            data = {
                'vocab' : self.vocab,
                'stoi' : self.stoi,
                'itos' : self.itos,
                'specials' : self.specials,
                'default_token' : self.default_token,
            }
            pickle.dump(data, f)

    def loadVocab(self, fname : str = 'vocab.pkl') -> None:
      with open(fname, 'rb') as f:
        data = pickle.load(f)

        self.default_token = data['default_token']
        self.specials = data['specials']
        self.vocab = data['vocab']
        self.stoi = data['stoi']
        self.itos = data['itos']