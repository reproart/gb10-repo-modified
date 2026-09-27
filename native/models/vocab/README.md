# Data files for profiles

`qwen3.8-flash-next-draft-vocab-65536.pt`: 65,536 token ids (a sorted Python
list, `torch.save`) for SGLang's `--speculative-token-map`, used by
`models/qwen3.8-flash-next.sh` with `DRAFT_VOCAB`. The MTP drafter then scores
only these rows of the lm_head instead of all 248,320; the target still
verifies every drafted token, so outputs do not change, only acceptance and
the draft's cost can.

The id set is `src/draft_vocab_65536.npy` (sha256 `6459e0fd…4b26d4`) from
[reproart/qwen3.8-Flash-DGX-AutoRound-modified](https://github.com/reproart/qwen3.8-Flash-DGX-AutoRound-modified),
where it came from upstream [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(`0c6df7e`; Apache-2.0, Copyright 2026 blazux): corpus frequency, BPE order
and all special tokens, weighted to English and code. CJK-heavy output loses
draft acceptance with it; that repo's `tools/build_draft_vocab.py` builds a
set from your own traffic (convert its .npy with
`torch.save([int(i) for i in numpy.load(path)], out)`).
