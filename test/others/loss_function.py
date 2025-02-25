# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/2/17 11:11
# @File: loss_function
# @Email: mlshenkai@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(
        self, logits: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.loss(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


class KDLoss1(nn.Module):
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(
        self,
        teacher_logits: torch.FloatTensor,
        student_logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:

        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

        teacher_mask = torch.isinf(teacher_logits)

        probs = teacher_probs * student_log_probs
        prob_probs = torch.masked_fill(probs, teacher_mask, 0)

        # sum
        x = torch.sum(prob_probs, dim=-1).view(-1)

        mask = (labels != self.IGNORE_INDEX).int()

        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class KDLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.IGNORE_INDEX = ignore_index

    def forward(
        self,
        teacher_logits: torch.FloatTensor,
        student_logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        -\sum_iP(i)*log(Q(i))
        其中：
        P(i) = softmax(teacher_logits)
        Q(i) = log(softmax(student_logits))


        Args:
            teacher_logits:
            student_logits:
            labels:

        Returns:

        """

        # 计算 P(i)
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        # 计算 log(Q(i))
        log_student_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

        # 去除teacher_logits 中inf的结果
        inf_mask = torch.isinf(teacher_logits)

        # 计算 P(i) * log(Q(i))
        probs = teacher_probs * log_student_probs

        # 填充 inf部分填充0
        probs = torch.masked_fill(probs, inf_mask, 0)

        # 计算\sum_i 部分

        x = torch.sum(probs, dim=-1).view(-1)

        # 计算mask 即ignore_index部分
        mask = (labels != self.IGNORE_INDEX).int()

        # mask填充
        x_hat = x * mask.view(-1)

        # 计算总的loss
        y = -torch.sum(x_hat, dim=0)

        # 计算平均值
        y_mean = y / torch.sum(mask.view(-1), dim=0)
        return y_mean


from typing import Dict, Tuple, Union, List


class RewardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: Dict[str, "torch.Tensor"], outputs: Dict[str, "torch.Tensor"]
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        attn_masks = inputs["attention_mask"]
        input_ids = inputs["input_ids"]
        bsz = input_ids.size(0) // 2
        chosen_masks, rejected_masks = torch.split(attn_masks, bsz, dim=0)
        
