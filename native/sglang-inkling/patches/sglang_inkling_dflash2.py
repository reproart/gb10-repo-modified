"""Minimal Inkling DFlash2 integration for the pinned official Inkling image.

No target math, sampling, or checkpoint weights are changed.
"""
from pathlib import Path
import hashlib, json

def apply():
    import sglang
    root=Path(sglang.__file__).resolve().parent
    changes=[]
    pending={}
    originals={}
    def read(rel):
        if rel not in pending:
            originals[rel]=pending[rel]=(root/rel).read_text()
        return pending[rel]
    def edit(rel, old, new):
        s=read(rel)
        if new in s:return
        assert s.count(old)==1,(rel,s.count(old),old[:100])
        pending[rel]=s.replace(old,new)
    def undo(rel, old, new):
        s=read(rel)
        if new in s:
            assert s.count(new)==1,(rel,new[:100])
            pending[rel]=s.replace(new,old)
    # Upgrade the earlier patch without retaining draft-owned target metadata.
    undo('srt/speculative/dflash_utils.py',
         '    is_mla: bool = False\n',
         '    is_mla: bool = False\n    capture_final_norm: bool = False\n')
    undo('srt/speculative/dflash_utils.py',
         '            if val < 0 or val >= target_num_layers:',
         '            if val < 0 or val > target_num_layers or (val == target_num_layers and not self.capture_final_norm):')
    undo('srt/speculative/dflash_utils.py',
         '        final_logit_softcapping=final_logit_softcapping,',
         '        final_logit_softcapping=final_logit_softcapping,\n'
         '        capture_final_norm=bool(dflash_cfg.get("capture_final_norm", False)),')
    alias = ('    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:\n'
             '        # Both drafts use native post-residual visit features and final norm.\n'
             '        self.set_dspark_layers_to_capture(layer_ids)\n\n')
    original = '    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:'
    undo('srt/models/inkling2.py', original, alias + original)
    edit('srt/models/inkling2.py', original,
         alias + '    def get_dflash_valid_vocab_size(self) -> int:\n'
         '        return int(self.text_config.vocab_size)\n\n' + original)
    edit('srt/configs/inkling2.py',
         '    @property\n    def num_layer_visits(self) -> int:',
         '    def resolve_dflash_capture_layers(self, layer_ids):\n'
         '        # Inkling features end with the post-final-norm state.\n'
         '        ids = list(layer_ids) if layer_ids is not None else []\n'
         '        if (not ids or ids != sorted(set(ids)) or ids[0] < 0\n'
         '                or ids[-1] != self.num_layer_visits):\n'
         '            raise ValueError("Inkling draft capture ids must be sorted, unique, "\n'
         '                             "and end at the final-norm layer visit")\n'
         '        return ids\n\n'
         '    @property\n    def num_layer_visits(self) -> int:')
    edit('srt/model_executor/model_runner_components/spec_aux_hidden_state.py',
         '        target_layer_ids = dflash_draft_config.resolve_target_layer_ids(\n'
         '            target_num_layers=int(target_num_layers),\n'
         '            draft_num_layers=int(draft_num_layers),\n'
         '        )',
         '        capture_resolver = getattr(model_config.hf_text_config,\n'
         '                                   "resolve_dflash_capture_layers", None)\n'
         '        if callable(capture_resolver):\n'
         '            target_layer_ids = capture_resolver(dflash_draft_config.target_layer_ids)\n'
         '        else:\n'
         '            target_layer_ids = dflash_draft_config.resolve_target_layer_ids(\n'
         '                target_num_layers=int(target_num_layers),\n'
         '                draft_num_layers=int(draft_num_layers),\n'
         '            )')
    edit('srt/models/dflash.py',
         '        target_layer_ids = draft_config.resolve_target_layer_ids(\n'
         '            target_num_layers=target_num_layers, draft_num_layers=num_layers\n'
         '        )',
         '        # Explicit feature ids are validated against the actual target\n'
         '        # by its capture adapter; here only the projection width is needed.\n'
         '        target_layer_ids = draft_config.target_layer_ids\n'
         '        if target_layer_ids is None:\n'
         '            target_layer_ids = draft_config.resolve_target_layer_ids(\n'
         '                target_num_layers=target_num_layers, draft_num_layers=num_layers\n'
         '            )')
    edit('srt/speculative/dflash_worker_v2.py',
         '        self.selector = self.draft_model.candidate_selector\n',
         '        self.selector = self.draft_model.candidate_selector\n'
         '        vocab_resolver = getattr(target_worker.model_runner.model,\n'
         '                                 "get_dflash_valid_vocab_size", None)\n'
         '        if self.selector is not None and callable(vocab_resolver):\n'
         '            self.draft_model.valid_vocab_size = vocab_resolver()\n')
    undo('srt/models/dflash.py',
         '        self.valid_vocab_size = None',
         '        self.valid_vocab_size = config.dflash_config.get("valid_vocab_size")')
    # The trained selector applies softcap after adding the correction. The
    # image predates that config field; preserve its default for other drafts.
    edit('srt/models/dflash.py',
         '        # The draft has no head of its own; the worker points this at the target\'s',
         '        self.selector_softcap_mode = config.dflash_config.get("selector_softcap_mode", "before_correction")\n'
         '        if self.selector_softcap_mode not in ("before_correction", "after_correction"):\n'
         '            raise ValueError(self.selector_softcap_mode)\n'
         '        self.candidate_selector.post_softcap = (draft_config.final_logit_softcapping\n'
         '            if self.selector_softcap_mode == "after_correction" else None)\n'
         '        self.valid_vocab_size = None\n'
         '        # The draft has no head of its own; the worker points this at the target\'s')
    edit('srt/models/dflash.py',
         '        if softcap is not None:\n            logits.div_(softcap).tanh_().mul_(softcap)',
         '        if softcap is not None and self.selector_softcap_mode == "before_correction":\n            logits.div_(softcap).tanh_().mul_(softcap)')
    edit('srt/models/dflash.py',
         '        return _score_edges(\n            predecessor_table=self.predecessor_codebook,',
         '        scores = _score_edges(\n            predecessor_table=self.predecessor_codebook,')
    edit('srt/models/dflash.py',
         '            top_k=self.top_k,\n        )\n\n    def sample_path(',
         '            top_k=self.top_k,\n        )\n'
         '        cap = getattr(self, "post_softcap", None)\n'
         '        return cap * torch.tanh(scores / cap) if cap else scores\n\n'
         '    def sample_path(')
    # Limit the draft candidate projection to the same real vocabulary used by
    # the target logits processor and training (201024 is padded storage only).
    edit('srt/models/dflash.py',
         '    with_partition: bool = False,\n    merged_unary_transform:',
         '    with_partition: bool = False,\n    valid_vocab_size: Optional[int] = None,\n    merged_unary_transform:')
    edit('srt/models/dflash.py',
         '    logits = _project_candidate_logits(\n        hidden, lm_head, num_org=num_org, use_quant_head=use_quant_head\n    )',
         '    if valid_vocab_size is not None:\n'
         '        if not tp:\n'
         '            raise ValueError("This Inkling benchmark integration is validated for TP1")\n'
         '        num_org = min(num_org, int(valid_vocab_size))\n'
         '    logits = _project_candidate_logits(\n        hidden, lm_head, num_org=num_org, use_quant_head=use_quant_head\n    )')
    edit('srt/models/dflash.py',
         '            hidden, self.lm_head, self.candidate_selector.top_k\n        )\n        return ids, self._transform_unary_logits(vals)',
         '            hidden, self.lm_head, self.candidate_selector.top_k,\n'
         '            valid_vocab_size=self.valid_vocab_size\n        )\n        return ids, self._transform_unary_logits(vals)')
    for rel,s in pending.items():
        compile(s, str(root/rel), 'exec')
    for rel,s in pending.items():
        before=originals[rel]
        if before != s:
            (root/rel).write_text(s)
            changes.append({'file':rel,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                            'after_sha256':hashlib.sha256(s.encode()).hexdigest()})
    return changes

if __name__=='__main__':
    print(json.dumps(apply(),indent=2))
