from collections import Counter

class Tokenizer:
    """
    A word-frequency-weighted Byte-Pair Encoding (BPE) tokenizer trainer.

    Instead of scanning the full raw text on every merge iteration, this
    tokenizer pre-splits the corpus into unique words and tracks each
    word's frequency separately. Pair counts during training are weighted
    by word frequency, so a pair inside a word that occurs 5,000 times
    contributes 5,000 to the count in one step, rather than requiring
    5,000 separate scans. Each unique word maintains its own current
    token sequence (self.word_tokens), which is updated independently as
    merges are learned. A '</w>' marker is appended to each word to mark
    word-end, preventing merges from spanning across word boundaries.

    Attributes:
        word_count (Counter): unique word -> frequency in the corpus.
        word_tokens (dict): unique word -> current list of tokens for
            that word (starts as individual characters + '</w>', and
            shrinks as merges are applied).
        vocabulary (list): all known token strings, including the
            initial character-level alphabet and every merged token
            learned during training.
        merges (list): ordered list of (pair, merged_token) learned
            during training. Order matters — this is the sequence
            later needed to tokenize new, unseen text consistently.
        num_merges (int): maximum number of merge operations to attempt.
        min_pair_freq (int): a pair must occur more than this many times
            (weighted by word frequency) to be merged; training stops
            early once no pair clears this bar.
    """
    def __init__(self,text:str,merges:int,min_pair_freq:int=2):
        """
        Build initial per-word token sequences and starting vocabulary
        from the raw corpus text.

        Args:
            text: the full training corpus.
            merges: maximum number of BPE merge operations to run.
            min_pair_freq: minimum number of appearance of a pair.
        """
        self.word_count = Counter(text.split())
        self.word_tokens = {}
        for word in self.word_count.keys():
            chars = list(word)
            chars.append('</w>') # marks word-end so merges can't cross word boundaries
            self.word_tokens[word] = chars

        # Starting vocabulary: every unique character in the corpus,
        # with the literal space character remapped to the '</w>' marker
        # so it matches how word-ends are represented in word_tokens.

        self.vocabulary = list(dict.fromkeys(text))
        self.vocabulary = ['</w>' if i==" " else i for i in self.vocabulary]

        self.merges = []
        self.num_merges = merges
        self.min_pair_freq = min_pair_freq
    def tokenize(self):
        """
        Run BPE training: repeatedly find the most frequent adjacent
        token pair (weighted by word frequency) across all words, and
        merge it everywhere it occurs, until num_merges is reached or
        no pair occurs frequently enough to be worth merging.

        Returns:
            (vocabulary, merges): the final vocabulary list, and the
            ordered list of learned merge rules.
        """
        for merge in range(self.num_merges):

            # --- Count weighted pair frequencies across all words ---
            token_counter = Counter()
            for word,chars in self.word_tokens.items():
                for i in range(len(chars)-1):
                    comb = (chars[i],chars[i+1])
                    # Weight by word frequency, not by 1: a pair inside a
                    # word occurring N times counts as N occurrences.

                    if comb in token_counter:
                        token_counter[comb] += self.word_count[word]
                    else:
                        token_counter[comb] = self.word_count[word]

            max_key = max(token_counter,key=token_counter.get)

            if (token_counter[max_key]>self.min_pair_freq):
                str_comb = "".join(max_key)
                # --- Apply the winning merge within every word's token list ---
                for word,chars in self.word_tokens.items():
                    chars_copy = []
                    k = 0   # start of the next unmerged slice to copy over
                    l = 0
                    while(l<len(chars)-1):
                        if (chars[l],chars[l+1]) == max_key:
                            chars_copy.extend(chars[k:l])   # copy untouched tokens before the match
                            chars_copy.append(str_comb)     # insert the merged token
                            k  = l+2                        # resume after the consumed pair
                            l += 1                          # extra step: skip past the pair's second element
                        l +=1

                    chars_copy.extend(chars[k:])            # copy whatever remains after the last match (or everything, if no match)
                    self.word_tokens[word] = chars_copy

                self.vocabulary.append(str_comb)
                self.merges.append((max_key,str_comb))

            else:
                # No pair occurs frequently enough to be worth merging further.
                break
        return self.vocabulary,self.merges