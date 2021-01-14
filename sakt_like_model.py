import gc
import psutil
import joblib
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#=====================================================================
# Riiidコンペで使用したbest sakt like model のtraining ファイル
# 通用のsaktモデルに使用する特徴量に加えてpartやelapsed timeを追加している。

TRAIN_SAMPLES = 290000
MAX_SEQ = 100
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3
TRAIN_BATCH_SIZE = 2048

DEBUG = False

if DEBUG == True:
    EPOCHS = 1
else:
    EPOCHS = 30


dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8','prior_question_elapsed_time':'float32'}
train_df = pd.read_feather('../input/riiid-train-data-multiple-formats/riiid_train.feather')[[
    'timestamp', 'user_id', 'content_id', 'content_type_id', 'answered_correctly','prior_question_elapsed_time'
]]
for col, dtype in dtypes.items():
    train_df[col] = train_df[col].astype(dtype)
train_df = train_df[train_df.content_type_id == False]

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df.part = questions_df.part.astype(np.int8)
train_df = pd.merge(train_df, questions_df[['question_id','part']],left_on='content_id',right_on='question_id',how="left")

del questions_df;gc.collect()

# Delete some rows due to memory limitage

gc.collect()
train_df.prior_question_elapsed_time /= 3600
prior_question_elapsed_time_mean = train_df.prior_question_elapsed_time.dropna().values.mean()
print(f"prior_quesiton_elapsed_time_mean:{prior_question_elapsed_time_mean}")
train_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean, inplace=True)
train_df.prior_question_elapsed_time.clip(lower=0,upper=16,inplace=True)
train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype('int16')
print(train_df.prior_question_elapsed_time.unique())
train_df = train_df.sort_values(['timestamp'], ascending=True)
train_df.reset_index(drop=True, inplace=True)
skills = train_df["content_id"].unique()
joblib.dump(skills, "skills.pkl.zip")
n_skill = len(skills)
n_p = train_df['part'].nunique()
print("number skills", len(skills))
print('Number Parts', n_p)

group = train_df[['user_id', 'content_id', 'answered_correctly','part','prior_question_elapsed_time']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values))
if DEBUG == False:
    group = group.loc[-int(len(group) * 0.7):]

joblib.dump(group, "group.pkl.zip")
del train_df
gc.collect()

if DEBUG:
    train_indexes = list(group.index)[:2000]
    valid_indexes = list(group.index)[-1000:]
else:
    train_indexes = list(group.index)[:TRAIN_SAMPLES]
    valid_indexes = list(group.index)[TRAIN_SAMPLES:]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]
del group, train_indexes, valid_indexes
print(len(train_group), len(valid_group))


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, n_p=7,min_samples=1, max_seq=128):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        self.n_p = 7
        self.n_time = 17

        self.user_ids = []
        for user_id in group.index:
            q, qa, p, e_time = group[user_id]
            if len(q) < min_samples:
                continue
            
            # Main Contribution
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial], p[:initial], e_time[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end], p[start:end], e_time[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa, p, e_time)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, p_, e_time_= self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        p = np.zeros(self.max_seq, dtype=int)
        e_time = np.zeros(self.max_seq, dtype=int)
        if seq_len == self.max_seq:
            q[:] = q_
            qa[:] = qa_
            p[:] = p_
            e_time[:] = e_time_
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            p[-seq_len:] = p_
            e_time[-seq_len:] = e_time_
        target_id = q[1:]
        label = qa[1:]
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill
        
        part = np.zeros(self.max_seq-1, dtype=int)
        part = p[:-1].copy()
        part += (qa[:-1] == 1) * self.n_p
        e_time = e_time[1:]
        
        return x, part, target_id, e_time,label
    
train_dataset = SAKTDataset(train_group, n_skill, min_samples=MIN_SAMPLES, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
valid_dataset = SAKTDataset(valid_group, n_skill, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill,n_p = 7,n_time = 17,max_seq=128, embed_dim=128, dropout_rate=0.2):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.n_p = n_p
        self.embed_dim = embed_dim
        self.n_time = 17
        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.p_embedding = nn.Embedding(2*n_p+1, embed_dim)
        self.t_embedding = nn.Embedding(n_time+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x, part, e_time,question_ids):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        
        p = self.p_embedding(part)
        
        elapsed_time = self.t_embedding(e_time)
        
        x = x + pos_x + p + elapsed_time

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight
    
def train_fn(model, dataloader, optimizer, scheduler, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        p = item[1].to(device).long()
        target_id = item[2].to(device).long()
        e_time = item[3].to(device).long()
        label = item[4].to(device).float()
        target_mask = (target_id != 0)

        optimizer.zero_grad()
        output, _, = model(x, p, e_time, target_id)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc


def valid_fn(model, dataloader, criterion, device="cpu"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        p = item[1].to(device).long()
        target_id = item[2].to(device).long()
        e_time = item[3].to(device).long()
        label = item[4].to(device).float()
        target_mask = (target_id != 0)

        output, _, = model(x, p, e_time, target_id)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)


best_auc = 0
max_steps = 3
step = 0
for epoch in range(EPOCHS):
    loss, acc, auc = train_fn(model, train_dataloader, optimizer, scheduler, criterion, device)
    print("epoch - {}/{} train: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, auc))
    loss, acc, auc = valid_fn(model, valid_dataloader, criterion, device)
    print("epoch - {}/{} valid: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, auc))
    if auc > best_auc:
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), "sakt_model.pt")
    else:
        step += 1
        if step >= max_steps:
            break
            
del train_dataset, valid_dataset

torch.save(model.state_dict(), "sakt_model_final.pt")


class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, n_p = 7, n_time = 17,max_seq=MAX_SEQ):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq
        self.n_p = n_p
        self.n_time = n_time

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        p = np.zeros(self.max_seq, dtype=int)
        e_time = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_, p_, e_time_= self.samples[user_id]
            
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
                p = p_[-self.max_seq:]
                e_time = e_time_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_    
                p[-seq_len:] = p_
                e_time[-seq_len:] = e_time_
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        part = np.zeros(self.max_seq-1, dtype=int)
        part = p[1:].copy()
        part += (qa[1:] ==1) * self.n_p
        
        questions = np.append(q[2:], [target_id])
        
        return x, part, e_time[1:], questions
    
    
import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()
group = joblib.load("group.pkl.zip")

model.eval()
prev_test_df = None

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df.part = questions_df.part.astype(np.int8)

for (test_df, sample_prediction_df) in tqdm(iter_test):
    if (prev_test_df is not None) & (psutil.virtual_memory().percent < 90):
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
        
        prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly','part', 'prior_question_elapsed_time']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values
        ))
        for prev_user_id in prev_group.index:
            if prev_user_id in group.index:
                group[prev_user_id] = (
                    np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:], 
                    np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:],
                    np.append(group[prev_user_id][2], prev_group[prev_user_id][2])[-MAX_SEQ:],
                    np.append(group[prev_user_id][3], prev_group[prev_user_id][3])[-MAX_SEQ:]
                )
 
            else:
                group[prev_user_id] = (
                    prev_group[prev_user_id][0], 
                    prev_group[prev_user_id][1],
                    prev_group[prev_user_id][2],
                    prev_group[prev_user_id][3]
                )
    test_df.prior_question_elapsed_time /= 3600
    test_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean, inplace=True)
    test_df.prior_question_elapsed_time.clip(lower=0,upper=16,inplace=True)
    test_df.prior_question_elapsed_time = test_df.prior_question_elapsed_time.astype('int16')
    test_df = pd.merge(test_df, questions_df[['question_id','part']], left_on='content_id', right_on='question_id', how="left")
    prev_test_df = test_df.copy()
    test_df = test_df[test_df.content_type_id == False]
    test_dataset = TestDataset(group, test_df, skills)
    test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)
    
    outs = []

    for item in tqdm(test_dataloader):
        x = item[0].to(device).long()
        p = item[1].to(device).long()
        e_time = item[2].to(device).long()
        target_id = item[3].to(device).long()

        with torch.no_grad():
            output, att_weight = model(x, p, e_time, target_id)
        outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())
        
    test_df['answered_correctly'] = outs
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
    
print('job done')