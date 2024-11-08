from hazm import Normalizer, word_tokenize
from typing import Mapping
from .vocab import Vocab
import torch

class AutoTokenizer:
    def __init__(
        self,
        tokenizer,
        vocab : Vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
    
    def __call__(
        self,
        input : tuple[str]
    ) -> Mapping[str, torch.Tensor]:
        tokens = [self.tokenizer(item) for item in input]
        
        max_length = len(max(tokens, key = len)) + 1
        bs = len(input)

        batch = torch.zeros((bs, max_length))

        for indx, item in enumerate(tokens):
            encoded = self.vocab.encode(item + ['[EOS]'])
            batch[indx, :len(encoded)] = torch.LongTensor(encoded)
        
        sos_axis = torch.zeros((bs, 1)).fill_(self.vocab['[SOS]']).long()
        batch = torch.cat((sos_axis, batch), dim = 1)

        padding_mask = (~(batch == self.vocab['[PAD]'])).int()
        
        return {
            "input_ids" : batch.long(),
            "padding_mask" : padding_mask,
        }
    
    def decode(
        self,
        input : tuple[int],
        ignore_special : bool = True
    ) -> str:
        input = input.cpu().numpy()
        tokens = self.vocab.decode(input, ignore_special)
        sentence = ""
        for i, token in enumerate(tokens):
            if i > 0 and token not in [
                ".",
                ",",
                "!",
                "?",
                ";",
                ":",
                "؟",
                "،"
            ]:
                sentence += " "
            sentence += token
        return sentence

def loadTokenizer(path : str) -> AutoTokenizer:
    vocab = Vocab()
    vocab.loadVocab(path)
    
    normalizer = Normalizer()
    tokenizer = lambda x : word_tokenize(normalizer.normalize(x))
    
    return AutoTokenizer(tokenizer, vocab)