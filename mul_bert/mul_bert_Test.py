import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置GPU ID
import torch
import bertRC_mALL_data_helps
from Mul_bert_Model import bertForRC_BERT
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
from transformers import BertConfig
import torch.nn as nn
from config import config

# gpu 设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

test_path = "../Datas/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
checkpoint_dir = config.checkpoint_dir

predictions = "predictions/"
max_sentence_length = 128
val_batch_size = 32
pretrained_weights = bertRC_mALL_data_helps.pretrained_weights
rt_tokenizer = bertRC_mALL_data_helps.tokenizer
cls_id = rt_tokenizer.cls_token_id
sep_id = rt_tokenizer.sep_token_id
pad_id = rt_tokenizer.pad_token_id
doler_id, jin_id = rt_tokenizer.additional_special_tokens_ids


config = BertConfig.from_pretrained(pretrained_weights)
config.num_labels = 19
config.output_hidden_states = True
config.output_attentions = True
model = bertForRC_BERT(config, rt_tokenizer, max_sentence_length=max_sentence_length, pretrained_weights=pretrained_weights)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)


# 数据
class RCDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


# 标签： 文字关系
label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


def eval():
    x_val, y_val, _ = bertRC_mALL_data_helps.load_data_and_labels(test_path)

    # 记载模型
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint["state_dict"], False)
    model.eval()

    val_dataset = RCDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    pred_y = []
    labels = []
    all_text = []
    with torch.no_grad():
        for index, (val_text, val_label) in enumerate(val_loader):
            labels.extend(val_label.tolist())
            all_text.extend(list(val_text))
            x_token = []
            x_mark_index_all = []
            for item in val_text:
                temp = rt_tokenizer.encode(item, add_special_tokens=False)
                while len(temp) < max_sentence_length:
                    temp.append(pad_id)
                temp_cup = list(enumerate(temp))
                cls_index = [index for index, value in temp_cup if value == cls_id]
                cls_index.append(0)
                doler_index = [index for index, value in temp_cup if value == doler_id]
                jin_index = [index for index, value in temp_cup if value == jin_id]
                sep_index = [index for index, value in temp_cup if value == sep_id]
                sep_index.append(0)
                x_mark_index = []
                x_mark_index.append(cls_index)
                x_mark_index.append(doler_index)
                x_mark_index.append(jin_index)
                x_mark_index.append(sep_index)
                x_mark_index_all.append(x_mark_index)
                x_token.append(temp)

            x_token = np.array(x_token)
            x_token = torch.tensor(x_token)
            x_token = x_token.to(device)

            out = model(x_token, x_mark_index_all)
            pred_y.extend(torch.max(out, 1)[1].tolist())

        f1_value = f1_score(labels, pred_y, average='macro')
        val_acc = np.mean(np.equal(labels, pred_y))
        print("Test(非官方): ACC: {}, F1: {}".format(val_acc, f1_value))

        prediction_path = os.path.join(predictions, "predictions.txt")
        truth_path = os.path.join(predictions, "ground_truths.txt")
        prediction_file = open(prediction_path, 'w')
        truth_file = open(truth_path, 'w')

        for i in range(len(pred_y)):
            prediction_file.write("{}\t{}\n".format(i, label2class[pred_y[i]]))
            truth_file.write("{}\t{}\n".format(i, label2class[labels[i]]))

        prediction_file.close()
        truth_file.close()

        # perl语言文件的源程序
        perl_path = os.path.join("../",
                                 "Datas",
                                 "semeval2010_task8_scorer-v1.2.pl")
        process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
        for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
            print(line)


if __name__ == "__main__":
    eval()