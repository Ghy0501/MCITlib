# LLaVA upstream

- Source: https://github.com/haotian-liu/LLaVA
- Commit: `31cceb832f2f403b2cbf822d0f8fe689d72fe219` (2023-11-27)
- Vendored content: `llava/` package and the upstream `LICENSE`.
- License: Apache-2.0, read from the upstream `LICENSE` at this commit.
- Local method-specific code: none. MCITlib methods and adapters live outside this directory.
- Patches: none currently. Any future generic hook must be stored under `patches/` and applied by `tools/check_third_party.py`.

Reproduce the original files (the repository is cloned outside this source tree):

```bash
git clone https://github.com/haotian-liu/LLaVA /tmp/mcitlib-upstream-LLaVA
git -C /tmp/mcitlib-upstream-LLaVA archive \
  31cceb832f2f403b2cbf822d0f8fe689d72fe219 llava LICENSE | \
  tar -x -C third_party/LLaVA
```
