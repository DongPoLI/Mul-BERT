import numpy as np
import pandas as pd
import re
import nltk
from transformers import BertTokenizer, BertModel
from config import config

pretrained_weights = config.pretrained_weights
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
special_tokens_dict = {'additional_special_tokens': ["$", "#"]}
print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
# 添加特殊Token, 使模型不会拆分， 用作标记使用
tokenizer.add_special_tokens(special_tokens_dict)
print(tokenizer.additional_special_tokens)
print(tokenizer.additional_special_tokens_ids)
print(tokenizer.sep_token)
print(tokenizer.sep_token_id)
print(tokenizer.cls_token_id)
print(tokenizer.cls_token)
print(tokenizer.pad_token_id)
print(tokenizer.mask_token)  # [MASK]
print(tokenizer.mask_token_id)  # 103


# 文字关系：标签 19;
class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

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


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=$#]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)  # ?
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)  # ?

    return text.strip()


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    labels = []  # 标签
    for idx in range(0, len(lines), 4):  # 处理每条句子
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        # 清除 原有的 # $, 特殊符号 不认为影响 句子意思
        sentence = sentence.replace('#', '')
        sentence = sentence.replace('$', '')

        sentence = sentence.replace('<e1>', ' $ ')
        sentence = sentence.replace('</e1>', ' $ ')
        sentence = sentence.replace('<e2>', ' # ')
        sentence = sentence.replace('</e2>', ' # ')

        sentence = clean_str(sentence)  # 对句子清洗一遍
        sentence = "[CLS] " + sentence + " [SEP]"  # 在句子开始 加入[CLS] or CLS ？ [CLS]:101; CLS:101

        # 暂不作处理
        # tokens = nltk.word_tokenize(sentence)
        tokens = tokenizer.tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        # sentence = " ".join(tokens)

        data.append([id, sentence, relation])

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [class2label[r] for r in df['relation']]
    x_text = df['sentence'].tolist()
    y = df['label'].tolist()

    x_text = np.array(x_text)
    y = np.array(y)
    # x_text = x_text.reshape(-1, 1)
    # y = y.reshape(-1, 1)

    return x_text, y, max_sentence_length  # 数据（句子），标签


if __name__ == "__main__":
    train_path = "/home/yons/PycharmProjects/stu_NLP/Attention-Based-BiLSTM-relation-extraction-master/" \
                 "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    load_data_and_labels(train_path, is_train=True)