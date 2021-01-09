import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import numpy as np


class bertForRC_BERT(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_sentence_length=128, is_cuda=True, num_labels=19,
                 pretrained_weights="bert-base-uncased"):
        super(bertForRC_BERT, self).__init__(config)
        self.num_labels = num_labels
        self.max_sentence_length = max_sentence_length

        self.tokenizer = tokenizer
        self.bertModel = BertModel.from_pretrained(pretrained_weights, config=config)  # , config=config
        self.is_cuda = is_cuda

        d = config.hidden_size
        self.test1_entity = nn.Linear(d, d * num_labels)
        self.test2_entity = nn.Linear(d, d * num_labels)

        nn.init.xavier_normal_(self.test1_entity.weight)
        nn.init.constant_(self.test1_entity.bias, 0.)
        nn.init.xavier_normal_(self.test2_entity.weight)
        nn.init.constant_(self.test2_entity.bias, 0.)

    def forward(self, x, x_mark_index_all, device):

        bertresult, rep, hs, atts = self.bertModel(x)
        batch_size = x.size()[0]  # size

        doler_result = []
        jin_result = []

        for i in range(batch_size):
            cls = x_mark_index_all[i][0]
            doler = x_mark_index_all[i][1]
            jin = x_mark_index_all[i][2]
            sep = x_mark_index_all[i][3]

            # $ entity $
            entity1 = torch.mean(bertresult[i, doler[0] + 1: doler[1], :], dim=0, keepdim=True)
            doler_result.append(entity1)

            # # entity #
            entity2 = torch.mean(bertresult[i, jin[0] + 1: jin[1], :], dim=0, keepdim=True)
            jin_result.append(entity2)


        # 拼接
        H_clr = bertresult[:, 0]
        H_doler = torch.cat(doler_result, 0)
        H_jin = torch.cat(jin_result, 0)

        test1 = H_doler + H_clr  #
        test2 = H_jin + H_clr
        test1 = test1.reshape(H_clr.shape[0], -1, H_clr.shape[-1])
        test2 = test2.reshape(H_clr.shape[0], -1, H_clr.shape[-1])

        test1 = self.test1_entity(test1)  # bs, 1, 768x19  F.dropout(torch.tanh(test1), 0.1)
        test2 = self.test2_entity(test2)  # bs, 1, 768x19  F.dropout(torch.tanh(test2), 0.1)
        test1 = test1.reshape(H_clr.shape[0], -1, test1.shape[-2], H_clr.shape[-1])  # bs, 19, 1, 768
        test2 = test2.reshape(H_clr.shape[0], -1, test1.shape[-2], H_clr.shape[-1])
        attn_score = torch.matmul(test1, test2.permute(0, 1, 3, 2))
        score = attn_score / np.sqrt(H_clr.shape[-1])  # np.sqrt(768)
        score = score.squeeze(-1)
        score = score.squeeze(-1)

        return score

