import os
import json
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import paddle
import paddlenlp
import paddle.nn.functional as F
from functools import partial
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
import paddle.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from paddlenlp.transformers import *

seed = 80471

init_from_ckpt = None

# 切换语言模型,加载预训练模型
# ernie-3.0-xbase-zh
# ernie-3.0-base-zh
# ernie-3.0-medium-zh
# bert-base-chinese

MODEL_NAME = 'ernie-3.0-base-zh'

max_seq_length = 352
train_batch_size = 16
valid_batch_size = 16
test_batch_size = 16

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 30

# 学习率预热比例
warmup_proportion = 0.1
# 学习率衰减比例
decay_proportion = 0.2

# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

# 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
max_grad_norm = 1.0

# 是否使用数据增强
enable_dataaug = False

# 是否开启对抗训练
enable_adversarial = False

# Rdrop Loss的超参数，若该值大于0.则加权使用R-drop loss
rdrop_coef = 0.2

# 损失函数设置
unbalance = 'Focal_loss'  # None , Focal_loss
focalloss_alpha = 0.5
focalloss_gamma = 2

# 训练结束后，存储模型参数
save_dir = "checkpoint/{}-{}".format(MODEL_NAME.replace('/', '-'), int(time.time()))


def read_jsonfile(file_name):
    data = []
    with open(file_name, encoding='utf-8') as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return data


train = pd.DataFrame(read_jsonfile("./data/train.json"))
test = pd.DataFrame(read_jsonfile("./data/testA.json"))

print("train size: {} \ntest size {}".format(len(train), len(test)))

train['text'] = [row['title'] + '[SEP]' + row['assignee'] + '[SEP]' + row['abstract'] for idx, row in train.iterrows()]
print(train['text'][0])
test['text'] = [row['title'] + '[SEP]' + row['assignee'] + '[SEP]' + row['abstract'] for idx, row in test.iterrows()]
train['concat_len'] = [len(row) for row in train['text']]
print(train['concat_len'])
test['concat_len'] = [len(row) for row in test['text']]

# 拼接后的文本长度分析
for rate in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print("训练数据中{:.0f}%的文本长度小于等于 {:.2f}".format(rate * 100, train['concat_len'].quantile(rate)))
plt.title("text length")
sns.distplot(train['concat_len'], bins=10, color='r')
sns.distplot(test['concat_len'], bins=10, color='g')
plt.show()

train_label = train["label_id"].unique()
# 查看标签label分布
plt.figure(figsize=(16, 8))
plt.title("label distribution")
sns.countplot(y='label_id', data=train)


# 创建数据迭代器
def read(df, istrain=True):
    if istrain:
        for _, data in df.iterrows():
            yield {
                "words": data['text'],
                "labels": data['label_id']
            }
    else:
        for _, data in df.iterrows():
            yield {
                "words": data['text'],
            }


# # 将生成器传入load_dataset
# 从样本中随机的按比例选取train data和testdata
train, valid = train_test_split(train, test_size=0.2, random_state=5)
# Flase 对应返回 MapDataset
train_ds = load_dataset(read, df=train, lazy=False)
valid_ds = load_dataset(read, df=valid, lazy=False)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 编码
def convert_example(example, tokenizer, max_seq_len=512, mode='train'):
    # 调用tokenizer的数据处理方法把文本转为id
    tokenized_input = tokenizer(example['words'], is_split_into_words=True, max_seq_len=max_seq_len)
    # print(tokenized_input)
    if mode == "test":
        return tokenized_input
    # 把意图标签转为数字id
    tokenized_input['labels'] = [example['labels']]
    return tokenized_input  # 字典形式，包含input_ids、token_type_ids、labels


# partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数
train_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='train',
    max_seq_len=max_seq_length)

valid_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='dev',
    max_seq_len=max_seq_length)

# 映射编码
train_ds.map(train_trans_func, lazy=False)
valid_ds.map(valid_trans_func, lazy=False)

# 初始化BatchSampler
# 用于生成指定随机数80471
np.random.seed(seed)

# 以batch的形式将数据进行划分，从而以batch的形式训练相应模型。
# 迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致。
# shuffle (bool) - 是否需要在生成样本下标时打乱顺序。默认值为False。
# batch_size (int) - 每mini-batch中包含的样本数。默认值为1。
train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=train_batch_size, shuffle=True)
valid_batch_sampler = paddle.io.BatchSampler(valid_ds, batch_size=valid_batch_size, shuffle=False)

# 定义batchify_fn
batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "labels": Stack(dtype="int32"),
}): fn(samples)


# 初始化DataLoader
def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


# DataLoader返回一个迭代器，该迭代器根据 batch_sampler 给定的顺序迭代一次给定的 dataset
# DataLoader支持单进程和多进程的数据加载方式，当 num_workers 大于0时，将使用多进程方式异步加载数据。
train_data_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_sampler=train_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True,
    worker_init_fn=_init_fn)
valid_data_loader = paddle.io.DataLoader(
    dataset=valid_ds,
    batch_sampler=valid_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True,
    worker_init_fn=_init_fn)

# 相同方式构造测试集
test_ds = load_dataset(read, df=test, istrain=False, lazy=False)

test_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    mode='test',
    max_seq_len=max_seq_length)

test_ds.map(test_trans_func, lazy=False)

test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=test_batch_size, shuffle=False)

test_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
}): fn(samples)

test_data_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_sampler=test_batch_sampler,
    collate_fn=test_batchify_fn,
    return_list=True)

labels = train['label_id'].unique()
# 下载预训练模型
pretrained_model = AutoModel.from_pretrained(MODEL_NAME)


class TextClassification(nn.Layer):
    def __init__(self, pretrained_model, num_classes, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ptm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ptm.config["hidden_size"],
                                    self.num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.ptm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


model = TextClassification(pretrained_model, num_classes=len(labels))

# 如果有预训练模型，则加载模型
if init_from_ckpt and os.path.isfile(init_from_ckpt):
    state_dict = paddle.load(init_from_ckpt)
    model.set_dict(state_dict)

# 训练总步数
# *30
max_steps = len(train_data_loader) * epochs

# 学习率衰减策略
#在开始的 warmup * total_steps 个Step中，学习率由0线性增加到learning_rate，然后再余弦衰减到0。
lr_scheduler = paddlenlp.transformers.CosineDecayWithWarmup(learning_rate=learning_rate, total_steps=max_steps, warmup=warmup_proportion)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler, #学习率
    parameters=model.parameters(), #指定优化器需要优化的参数
    weight_decay=weight_decay, # 权重衰减系数，类似模型正则项策略，避免模型过拟合0.01
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm)) # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小


class FGSM:
    def __init__(self, model, epsilon=0.1, emb_name='word_embeddings'):
        self.model = (model.module if hasattr(model, "module") else model)
        self.eps = epsilon
        self.emb_name = emb_name
        self.backup = {}

    # only attack word embedding
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and self.emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self):
        for name, para in self.model.named_parameters():
            if para.stop_gradient and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


class FGM:
    def __init__(self, model, epsilon=1., emb_name='word_embeddings'):
        self.model = (model.module if hasattr(model, "module") else model)
        self.eps = epsilon
        self.emb_name = emb_name
        self.backup = {}

    # only attack embedding
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.stop_gradient and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = paddle.norm(param.grad)
                if norm and not paddle.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, para in self.model.named_parameters():
            if para.stop_gradient and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


# 对抗训练
if enable_adversarial:
    # adv = FGSM(model=model,epsilon=1e-6,emb_name='word_embeddings')
    adv = FGM(model=model, epsilon=1e-6, emb_name='word_embeddings')


class FocalLoss(paddle.nn.Layer):
    def __init__(self, alpha=0.5, gamma=2, num_classes=3, weight=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight if weight is not None else paddle.to_tensor(np.array([1.] * num_classes), dtype='float32')
        self.ce_fn = paddle.nn.CrossEntropyLoss(
            weight=self.weight, soft_label=False, ignore_index=ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = paddle.exp(logpt)
        # alpha 可以抑制正负样本的数量失衡，通过gamma可以控制简单/难区分样本数量失衡。
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


# --修改损失函数
# 损失函数设置
# unbalance = 'Focal_loss' #  None , Focal_loss
# focalloss_alpha = 0.5
# focalloss_gamma = 2
if unbalance == "Focal_loss":
    criterion = FocalLoss(
        alpha=focalloss_alpha,
        gamma=focalloss_gamma,
        num_classes=len(labels))
else:
    # 交叉熵损失
    criterion = paddle.nn.loss.CrossEntropyLoss()


# 定义模型训练验证评估函数
# 停止求导 这样的注释的效果是在我们进行模型预测的时候可以获取到较为稳定的预测结果。
@paddle.no_grad()
def evaluate(model, data_loader):
    model.eval()

    real_s = []
    pred_s = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)

        probs = F.softmax(logits, axis=1)
        pred_s.extend(probs.argmax(axis=1).numpy())
        real_s.extend(labels.reshape([-1]).numpy())
    score = f1_score(y_pred=pred_s, y_true=real_s, average='macro')

    return score  # F1-score


def do_train(model, train_data_loader, valid_data_loader, criterion, optimizer, lr_scheduler, rdrop_coef,
             enable_adversarial=False):
    model.train()  # 启用 batch  normalization和 dropout，保证 BN 层能够用到 每一批数据 的均值和方
    max_f1_score = 0
    if rdrop_coef > 0:
        rdrop_loss = paddlenlp.losses.RDropLoss()  # Rdrop Loss的超参数，若该值大于0.则加权使用R-drop loss
    for epoch in range(1, epochs + 1):
        with tqdm(total=len(train_data_loader)) as pbar:
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                logits = model(input_ids, token_type_ids)
                # --修改loss / 当rdrop_coef大于0时启用Rdrop
                if rdrop_coef > 0:
                    logits_2 = model(input_ids=input_ids, token_type_ids=token_type_ids)
                    ce_loss = (criterion(logits, labels).mean() + criterion(logits_2, labels).mean()) * 0.5
                    kl_loss = rdrop_loss(logits, logits_2)
                    loss = ce_loss + kl_loss * rdrop_coef
                else:
                    loss = criterion(logits, labels).mean()
                loss.backward()
                # 对抗训练
                if enable_adversarial:
                    adv.attack()  # 在 embedding 上添加对抗扰动
                    adv_logits = model(input_ids, token_type_ids)
                    adv_loss = criterion(adv_logits, labels).mean()
                    adv_loss.backward()  # 反向传播，并在正常的 grad 基础上，累加对抗训练的梯度
                    adv.restore()  # 恢复 embedding 参数
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                pbar.set_postfix({'loss': '%.5f' % (loss.numpy())})
                pbar.update(1)
        eval_f1_score = evaluate(model, valid_data_loader)
        print("Epoch: %d, eval_f1_score: %.5f" % (epoch, eval_f1_score))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  ## 递归创建

        print("Epoch: %d, eval_f1_score: %.5f" % (epoch, eval_f1_score),
              file=open(save_dir + '/best_model_log.txt', 'a'))

        if eval_f1_score >= max_f1_score:
            max_f1_score = eval_f1_score
            save_param_path = os.path.join(save_dir, 'best_model.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)
    save_param_path = os.path.join(save_dir, 'last_model.pdparams')
    paddle.save(model.state_dict(), save_param_path)

#模型、训练集、验证集、损失函数、优化器、学习率衰减策略、rdrop_coef = 0.2、对抗训练
do_train(model, train_data_loader, valid_data_loader, criterion, optimizer, lr_scheduler, rdrop_coef,
         enable_adversarial)


# 预测阶段
def do_sample_predict(model, data_loader, is_prob=False):
    model.eval()
    preds = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        preds.extend(probs.argmax(axis=1).numpy())
    if is_prob:
        return probs
    return preds


# 读取最佳模型
state_dict = paddle.load(os.path.join(save_dir, 'best_model.pdparams'))
model.load_dict(state_dict)

# 预测
pred_score = do_sample_predict(model, test_data_loader)

# 生成提交结果文件
sumbit = pd.DataFrame({"id": test["id"]})
sumbit["label"] = pred_score
file_name = "sumbit_{}.csv".format(save_dir.split("/")[1])
sumbit.to_csv(file_name, index=False)
print("生成提交文件{}".format(file_name))
