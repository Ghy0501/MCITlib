# peft upstream

- Source: https://github.com/huggingface/peft
- Tag/commit: `v0.4.0`
- Vendored content: `src/peft`, moved to `peft/` at this directory's root, plus the upstream `LICENSE`.
- License: Apache-2.0, read from the upstream `LICENSE` at `v0.4.0`.
- Local method-specific code: none. MCITlib-owned tuners live in `mcitlib/tuners/`.
- DoRA: not included in v4 (NB-1); no DoRA patch is applied.
- Patches: none currently. Any future generic hook must be stored under `patches/` and applied by `tools/check_third_party.py`.

Reproduce the original files:

```bash
git clone https://github.com/huggingface/peft /tmp/mcitlib-upstream-peft
git -C /tmp/mcitlib-upstream-peft archive v0.4.0 src/peft LICENSE | \
  tar -x -C third_party/peft
mv third_party/peft/src/peft third_party/peft/peft
rmdir third_party/peft/src
```
