# Byte-Pair Encoding (BPE) — From Scratch

A minimal, from-scratch Python implementation of the Byte-Pair Encoding (BPE) algorithm used to train subword tokenizers, similar in spirit to the tokenizers behind modern LLMs (GPT-2 style).

This project implements the **training** side of BPE: given a text corpus, it learns a vocabulary of subword units and an ordered list of merge rules by iteratively combining the most frequent adjacent pair of tokens.

## How it works

BPE starts by treating every character as its own token, then repeatedly finds the most frequent adjacent pair of tokens across the corpus and merges it into a single new token. Over many iterations, this builds up a vocabulary containing frequent whole words and subword fragments, in addition to individual characters — allowing rare or unseen words to still be represented as a sequence of known subword pieces.

This implementation is **word-frequency weighted** for efficiency:

- The corpus is split into unique words, and each word's frequency is counted once.
- Each unique word maintains its own token sequence (starting as individual characters, with a `</w>` marker appended to mark word-end).
- When counting adjacent pairs, each word's pairs are weighted by how often that word occurs in the corpus — so a pair inside a word occurring 5,000 times contributes 5,000 to the pair count in a single step, rather than requiring the word to be scanned 5,000 separate times.
- Merges are applied independently within each word's token sequence, and the `</w>` marker prevents merges from spanning across word boundaries.

This keeps training tractable on larger corpora by avoiding redundant work on repeated words.

## Usage

```bash
python tokenizer.py
```

By default, `tokenizer.py` reads a text file (e.g. `input.txt`) from the working directory, trains a BPE tokenizer on it for a configurable number of merges, and prints the learned vocabulary and merge rules.

```python
from tokenizer import Tokenizer

with open("input.txt", "r", encoding="utf-8-sig") as f:
    text = f.read()

tokenizer = Tokenizer(text, merges=1000,min_pair_freq=3)
vocabulary, merges = tokenizer.tokenize()

print(vocabulary)  # all learned tokens, from single characters to merged subwords
print(merges)      # ordered list of (pair, merged_token) rules learned during training
```

### Parameters

- `text` — the raw training corpus (a string).
- `merges` — the maximum number of merge operations to perform. Training may stop earlier if no pair occurs frequently enough to be worth merging (controlled internally by `min_pair_freq`).

### Output

- `vocabulary` — a list of every token the tokenizer knows, from the initial character-level alphabet up through every learned subword/whole-word merge.
- `merges` — the ordered list of merge rules `(pair, merged_token)` learned during training. The order matters: it reflects the sequence in which merges must be replayed to consistently tokenize new text.

## Current limitations / roadmap

This repo currently implements **training only**. Planned/possible next steps:

- **`encode(text)`** — apply the learned, ordered merge rules to tokenize new, unseen text.
- **`decode(tokens)`** — reconstruct original text from a sequence of tokens.
- **Punctuation-aware pre-tokenization** — currently splitting is whitespace-only (`text.split()`), so punctuation stays attached to words (e.g. `"lord,"` and `"lord"` are learned as separate units). A regex-based word/punctuation splitter would reduce redundant vocabulary entries.
- **Incremental pair-count updates** — pair counts are currently recomputed from scratch each merge iteration; updating counts only near the positions that changed after a merge would speed up training on larger corpora.

## Example

Trained on the Tiny Shakespeare dataset, the learned vocabulary progresses from single characters up through common subwords and whole words, e.g.:

```
'e</w>', 'th', 'the', 'and</w>', 'ing</w>', 'you</w>', 'KING</w>', 'QUEEN</w>', ...
```

demonstrating the algorithm correctly discovers frequent English morphemes and whole-word units purely from corpus statistics.

## Requirements

- Python 3.x
- No external dependencies beyond the standard library (`collections.Counter`)
