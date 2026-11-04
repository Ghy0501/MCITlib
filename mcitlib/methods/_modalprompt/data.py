# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""ModalPrompt's prompt-aware modality-length calculation."""
from __future__ import annotations

from mcitlib.backbones.llava.standard_train import LazySupervisedDataset


class ModalPromptLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, *args, prefix_len: int, cur_task: int, transfer_num: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_len = prefix_len
        self.cur_task = cur_task
        self.transfer_num = transfer_num

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len += min(self.cur_task, self.transfer_num) * self.prefix_len
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list
