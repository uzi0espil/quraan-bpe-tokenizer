from collections import Counter
from typing import Dict, List, Tuple, Optional
import regex as re
import unicodedata


class BPETokenizerV3(object):
    """This is an extended class that only adds the ability to split the text using regex and train on each chunk"""

    # since utf-8 is fixed, then we have an initial 256 encoded characters
    INITIAL_VOCAB_SIZE = 256
    # default pattern of GPT4
    DEFAULT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

    def __init__(self, vocab_size, pattern=None):
        if vocab_size < self.INITIAL_VOCAB_SIZE:
            raise ValueError(f"Vocab size should be higher than {self.INITIAL_VOCAB_SIZE}")

        self.pattern = pattern if pattern else self.DEFAULT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.num_merges = vocab_size - self.INITIAL_VOCAB_SIZE

        # attributes
        self._merges: Optional[Dict[Tuple[int, int], int]] = None
        self.vocab_: Optional[Dict[int, bytes]] = None
        self.special_tokens_: Dict[str, int] = dict()
        self.inverse_special_tokens: Dict[int, str] = dict()

    @property
    def is_fit(self) -> bool:
        """Instance is only considered fit only if we fitted at least once."""
        return self._merges is not None and self.vocab_ is not None

    def __len__(self):
        return len(self.vocab_) if self.vocab_ else 0

    @staticmethod
    def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Give you how many a pair appeared in list of bytes ordered in descendent manner."""
        pairs = [(p1, p2) for p1, p2 in zip(ids[:-1], ids[1:])]
        counts = Counter(pairs)
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        """Replace `pair` of ids with a new `id` within the list of `ids`"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def fit(self, text, special_tokens: Optional[List[str]] = None):
        """Fit the tokenizer on a text, the vocab is reset everytime fit is invoked."""
        # clean text from any special tokens:
        if special_tokens:
            escaped = [re.escape(tok) for tok in special_tokens]
            remove_pat = re.compile("|".join(escaped))
            text = remove_pat.sub("", text)

        # reset the trained parameters
        self._merges = dict()
        self.vocab_ = {idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)}

        # break text into chunks and convert each chunk to bytes
        text_chunks = re.findall(self.compiled_pattern, text)
        tokens = [list(bytearray(ch, "utf-8")) for ch in text_chunks]

        # standard merge loop to replace the top pair with new id
        idx = 0
        for i in range(self.num_merges):
            stats = Counter()
            [stats.update(self._get_stats(chunk_ids)) for chunk_ids in tokens]
            if len(stats) == 0:  # this happens if all words have exactly 1 id in the text.
                print(f"Reached the maximum number of merge in {i} given the regex pattern")
                break
            top_pair = max(stats, key=stats.get)
            idx = self.INITIAL_VOCAB_SIZE + i
            tokens = [self._merge(chunk_ids, top_pair, idx) for chunk_ids in tokens]
            self._merges[top_pair] = idx
            self.vocab_[idx] = self.vocab_[top_pair[0]] + self.vocab_[top_pair[1]]

        # register the special tokens starting from the last generated index.
        if special_tokens:
            self.register_special_tokens(special_tokens, idx + 1)

    def encode(self, text, allowed_special: bool = True):
        """Encode text into tokens given the trained vocab"""
        if not self.is_fit:
            raise Exception("Tokenizer isn't trained!")

        if not self.special_tokens_ or not allowed_special:
            return self.encode_ordinary(text)

        # split special special tokens from text.
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens_.keys()) + ")"
        text_chunks = re.split(special_pattern, text)

        # get the id from special token if found, otherwise encode the text normally.
        ids = []
        for text_chunk in text_chunks:
            if text_chunk in self.special_tokens_:
                ids.append(self.special_tokens_[text_chunk])
            else:
                ids.extend(self.encode_ordinary(text_chunk))
        return ids

    def register_special_tokens(self, special_tokens, starting_idx):
        """method to register special tokens"""
        for i, s_token in enumerate(special_tokens):
            current_index = starting_idx + i
            self.special_tokens_[s_token] = current_index
            self.vocab_[current_index] = s_token.encode("utf-8")

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        if not self.is_fit:
            raise Exception("Tokenizer isn't trained!")

        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        return [i for chunk in text_chunks for i in self._encode(chunk)]

    def _encode(self, text):
        """Encode text into tokens given the trained vocab"""
        tokens = list(bytearray(text, "utf-8"))
        while len(text) > 2:  # if the text has one character, then we only encode it with utf-8
            stats = self._get_stats(tokens)
            if len(stats) == 0:
                break
            pair = min(stats, key=lambda p: self._merges.get(p, float("inf")))
            if pair not in self._merges:
                break  # nothing to merge and basically `pair` contain a random pair that has a value of inf.
            idx = self._merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        """Decode ids back into text."""
        if not self.is_fit:
            raise Exception("Tokenizer isn't trained!")

        # get from vocab, if not found, then it is most probably a special token, otherwise it will crash.
        tokens = b"".join(self.vocab_.get(idx) for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def _render_token(self, t: bytes) -> str:
        """pretty print a token, escaping control characters"""
        s = t.decode('utf-8', errors='replace')
        chars = []
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)  # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}")  # escape
        return "".join(chars)

    def save(self, file_prefix, save_vocab=True):
        """Save the BPE for later usage.
        If `save_vocab` is true, a vocab file is saved containing all decoded vocabs with their index.
        Inspired by minbpe"""
        # write the model: to be used in load() later
        model_file = file_prefix + ".bpe" if not file_prefix.endswith(".bpe") else file_prefix
        with open(model_file, 'w') as f:
            # write pattern
            f.write(f"{self.pattern}\n")
            # write special tokens
            f.write(" ".join(self.special_tokens_.keys()))
            f.write("\n")
            # write merges
            [f.write(f"{p1} {p2}\n") for (p1, p2) in self._merges]

        # for verbosity, inspired from minbpe
        if save_vocab:
            # write the vocab: for the human to look at
            vocab_file = file_prefix + ".vocab"
            inverted_merges = {idx: pair for pair, idx in self._merges.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                for idx, token in self.vocab_.items():
                    # note: many tokens may be partial utf-8 sequences
                    # and cannot be decoded into valid strings. Here we're using
                    # errors='replace' to replace them with the replacement char �.
                    # this also means that we couldn't possibly use .vocab in load()
                    # because decoding in this way is a lossy operation!
                    s = self._render_token(token)
                    # find the children of this token, if any
                    if idx in inverted_merges:
                        # if this token has children, render it nicely as a merge
                        idx0, idx1 = inverted_merges[idx]
                        s0 = self._render_token(self.vocab_[idx0])
                        s1 = self._render_token(self.vocab_[idx1])
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        # otherwise this is leaf token, just print it
                        # (this should just be the first 256 tokens, the bytes)
                        f.write(f"[{s}] {idx}\n")

    @classmethod
    def load(cls, path):
        """Load instance giving the export bpe file."""
        assert path.endswith(".bpe")

        merge_idx = cls.INITIAL_VOCAB_SIZE
        merges = dict()
        with open(path, 'r', encoding="utf-8") as f:
            pattern = f.readline().strip()
            specials = f.readline().strip().split()
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = merge_idx
                merge_idx += 1
        ins = cls(merge_idx, pattern=pattern)
        ins._merges = merges
        ins.vocab_ = {idx: bytes([idx]) for idx in range(cls.INITIAL_VOCAB_SIZE)}
        for (p0, p1), idx in ins._merges.items():
            ins.vocab_[idx] = ins.vocab_[p0] + ins.vocab_[p1]
        ins.register_special_tokens(specials, starting_idx=merge_idx)
        return ins
