import psutil
import joblib

import numpy as np
import pandas as pd
from collections import defaultdict
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import dill
import riiideducation

#=====================================================================
# Riiidコンペで使用した sakt like modelとlgb modelのアンサンブル 




def pickle_load(path):
    with open(path, mode='rb') as f:
        data = dill.load(f)
        return data

# ===========================Constants=============================
DEBUG = False

question_cols = ['question_id','part','tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']

FEATURES = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'part', 'answered_correctly_u_avg','answered_correctly_h_avg', 'elapsed_time_u_avg', 'explanation_u_avg',
                'answered_correctly_q_avg', 'elapsed_time_q_avg', 'explanation_q_avg', 'answered_correctly_uq_count', 
                'timestamp_u_recency_1', 'timestamp_u_recency_2', 'timestamp_u_recency_3', 
                'timestamp_u_incorrect_recency', 'answered_correctly_up_avg',
                'tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']


TARGET = 'answered_correctly'
question_file = '../input/exp-013/questions_df.pkl'

MAX_SEQ = 100
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3

TRAIN_BATCH_SIZE = 2048

skills = joblib.load("../input/exp-040/skills.pkl.zip")
n_skill = len(skills)
n_p = 7

print("number skills", len(skills))
print('Number Parts', n_p)




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
    
    

 # Funcion for user stats with loops
def add_features(df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                 timestamp_u, timestamp_u_incorrect, 
                 answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, answered_correctly_up_count,answered_correctly_up_sum, update = True):
    # -----------------------------------------------------------------------
    # Client features
    answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
    explanation_u_avg = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)
#     timestamp_u_recency_4 = np.zeros(len(df), dtype = np.float32)
#     timestamp_u_recency_5 = np.zeros(len(df), dtype = np.float32)
    
    
    timestamp_u_incorrect_recency = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Question features
    answered_correctly_q_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
    explanation_q_avg = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Harmonic Mean Feature
    answered_correctly_h_avg = np.zeros(len(df),dtype=np.float32)
    # Client Question feature
    answered_correctly_uq_count= np.zeros(len(df),dtype=np.int32)
    # -----------------------------------------------------------------------
    # Client Part feature
    answered_correctly_up_avg = np.zeros(len(df), dtype=np.float32)
    # Time differenct features
    time_lag_mean = np.zeros(len(df),dtype=np.float32)
    time_lag_std = np.zeros(len(df), dtype=np.float32)
    # time_lag_skew = np.zeros(len(df),dtype=np.float32)

    for num, row in enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','part']].values):
        
        # Client features assignation
        # ------------------------------------------------------------------
        if answered_correctly_u_count[row[0]] != 0:
            answered_correctly_u_avg[num] = answered_correctly_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            elapsed_time_u_avg[num] = elapsed_time_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            explanation_u_avg[num] = explanation_u_sum[row[0]] / answered_correctly_u_count[row[0]]
        else:
            answered_correctly_u_avg[num] = np.nan
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan
            
        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]

        if len(timestamp_u_incorrect[row[0]]) == 0:
            timestamp_u_incorrect_recency[num] = np.nan
        else:
            timestamp_u_incorrect_recency[num] = row[5] - timestamp_u_incorrect[row[0]][0]
            
        # ------------------------------------------------------------------
        # Question features assignation
        if answered_correctly_q_count[row[2]] != 0:
            answered_correctly_q_avg[num] = answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        else:
            answered_correctly_q_avg[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan
        # ------------------------------------------------------------------
        # User Question feature assignation
        answered_correctly_uq_count[num] = answered_correctly_uq[row[0]][row[2]]
        # ------------------------------------------------------------------
        # Harmonic Mean Feature assignation
        answered_correctly_h_avg[num] = 2 * (answered_correctly_u_avg[num] * answered_correctly_q_avg[num])/(answered_correctly_u_avg[num] + answered_correctly_q_avg[num])
        # ------------------------------------------------------------------
        # Client Part feature assignation
        if answered_correctly_up_count[row[0]][row[6]] != 0:
            answered_correctly_up_avg[num] = answered_correctly_up_sum[row[0]][row[6]]/answered_correctly_up_count[row[0]][row[6]]
        else:
            answered_correctly_up_avg[num] = np.nan
        # ------------------------------------------------------------------
        #=============DICTIONARY UPDATE Part======================================== 
        # Client features updates
        elapsed_time_u_sum[row[0]] += row[3]
        explanation_u_sum[row[0]] += int(row[4])
        if len(timestamp_u[row[0]]) == 3:
            timestamp_u[row[0]].pop(0)
            timestamp_u[row[0]].append(row[5])
        else:
            timestamp_u[row[0]].append(row[5])
        
        # ------------------------------------------------------------------
        # Question features updates
        elapsed_time_q_sum[row[2]] += row[3]
        explanation_q_sum[row[2]] += int(row[4])
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_count[row[0]] += 1
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])
            
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_count[row[2]] += 1
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------
            # Client Question feature updates
            answered_correctly_uq[row[0]][row[2]] += 1
            # Client Part feature update
            answered_correctly_up_count[row[0]][row[6]] += 1
            answered_correctly_up_sum[row[0]][row[6]] += row[1]
            
             
            
    user_df = pd.DataFrame({'answered_correctly_u_avg': answered_correctly_u_avg, 'elapsed_time_u_avg': elapsed_time_u_avg, 'explanation_u_avg': explanation_u_avg, 
                            'answered_correctly_q_avg': answered_correctly_q_avg, 'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg,
                            'answered_correctly_h_avg': answered_correctly_h_avg, 'answered_correctly_up_avg':answered_correctly_up_avg,'answered_correctly_uq_count': answered_correctly_uq_count, 
                            'timestamp_u_recency_1': timestamp_u_recency_1, 'timestamp_u_recency_2': timestamp_u_recency_2,
                            'timestamp_u_recency_3': timestamp_u_recency_3, 'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency,
                            })#'timestamp_u_recency_4': timestamp_u_recency_4,'timestamp_u_recency_5': timestamp_u_recency_5,  
    
    df = pd.concat([df, user_df], axis = 1)
    return df


def create_features_dicts(df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                 timestamp_u, timestamp_u_incorrect, 
                 answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, answered_correctly_up_count,answered_correctly_up_sum, update = True):
   
    for num, row in enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','part']].values):

        #=============DICTIONARY UPDATE Part======================================== 
        # Client features updates
        elapsed_time_u_sum[row[0]] += row[3]
        explanation_u_sum[row[0]] += int(row[4])
        if len(timestamp_u[row[0]]) == 3:
            timestamp_u[row[0]].pop(0)
            timestamp_u[row[0]].append(row[5])
        else:
            timestamp_u[row[0]].append(row[5])
        
        # ------------------------------------------------------------------
        # Question features updates
        elapsed_time_q_sum[row[2]] += row[3]
        explanation_q_sum[row[2]] += int(row[4])
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_count[row[0]] += 1
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])
            
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_count[row[2]] += 1
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------
            # Client Question feature updates
            answered_correctly_uq[row[0]][row[2]] += 1
            # Client Part feature update
            answered_correctly_up_count[row[0]][row[6]] += 1
            answered_correctly_up_sum[row[0]][row[6]] += row[1]
            
    return 

   
    
def update_features(df, answered_correctly_u_sum,answered_correctly_u_count, answered_correctly_q_sum, answered_correctly_q_count,timestamp_u_incorrect, answered_correctly_uq, answered_correctly_up_count, answered_correctly_up_sum):
    for row in df[['user_id', 'answered_correctly', 'content_id', 'content_type_id', 'timestamp','part']].values:
        if row[3] == 0:
            # ------------------------------------------------------------------
            # Count feature updates
            answered_correctly_u_count[row[0]] += 1
            answered_correctly_q_count[row[2]] += 1
            answered_correctly_uq[row[0]][row[2]] += 1
            
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[4])
                else:
                    timestamp_u_incorrect[row[0]].append(row[4])
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------    
            # Client Part feature update
            answered_correctly_up_count[row[0]][row[5]] += 1
            answered_correctly_up_sum[row[0]][row[5]] += row[1]
    return



def read_and_preprocess(feature_engineering = False):
    
    train_pickle = '../input/riiid-cross-validation-files/cv1_train.pickle'
    valid_pickle = '../input/riiid-cross-validation-files/cv1_valid.pickle'
    
    # Read data
    feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']
    if DEBUG:
        train = pd.read_csv("../input/riiid-test-answer-prediction/train.csv",nrows=200000)[feld_needed]
        valid = train.iloc[-10000:]
        train = train.iloc[:-10000]
    else:
        train = pd.read_pickle(train_pickle)[feld_needed]
        valid = pd.read_pickle(valid_pickle)[feld_needed]
    # Delete some trianing data to don't have ram problems   
    if feature_engineering:
        if DEBUG == False:
            train = train.iloc[-40000000:]
            
    # Filter by content_type_id to discard lectures
    train = train.loc[train.content_type_id == False].reset_index(drop = True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)
    
    # Changing dtype to avoid lightgbm error
    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
    valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')
    
    # Fill prior question elapsed time with the mean
    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    
    # Merge with question dataframe
    questions_df = pd.read_pickle(question_file)
    question_cols = ['question_id','part','tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']
    train = pd.merge(train, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    valid = pd.merge(valid, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    del questions_df;gc.collect()
    
    # Client dictionaries
    answered_correctly_u_count = defaultdict(int)
    answered_correctly_u_sum = defaultdict(int)
    elapsed_time_u_sum = defaultdict(int)
    explanation_u_sum = defaultdict(int)
    timestamp_u = defaultdict(list)
    timestamp_u_incorrect = defaultdict(list)
   
    
    # Question dictionaries
    answered_correctly_q_count = defaultdict(int)
    answered_correctly_q_sum = defaultdict(int)
    elapsed_time_q_sum = defaultdict(int)
    explanation_q_sum = defaultdict(int)
    
    # Client Question dictionary
    answered_correctly_uq = defaultdict(lambda: defaultdict(int))
    
    
    # Client Part dictionary 
    answered_correctly_up_count = defaultdict(lambda:defaultdict(int))
    answered_correctly_up_sum = defaultdict(lambda:defaultdict(int))
    
    print('User feature calculation started...')
    create_features_dicts(train, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u,timestamp_u_incorrect, 
                         answered_correctly_q_count, answered_correctly_q_sum, 
                         elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq,
                         answered_correctly_up_count, answered_correctly_up_sum, update=True
                         )
    del train;gc.collect()
    create_features_dicts(valid, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u, timestamp_u_incorrect, 
                         answered_correctly_q_count, answered_correctly_q_sum, 
                         elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq,
                         answered_correctly_up_count, answered_correctly_up_sum, update=True
                         )
    del valid;gc.collect()
    print('User feature calculation completed...')
    
    features_dicts = {
        'answered_correctly_u_count': answered_correctly_u_count,
        'answered_correctly_u_sum': answered_correctly_u_sum,
        'elapsed_time_u_sum': elapsed_time_u_sum,
        'explanation_u_sum': explanation_u_sum,
        'answered_correctly_q_count': answered_correctly_q_count,
        'answered_correctly_q_sum': answered_correctly_q_sum,
        'elapsed_time_q_sum': elapsed_time_q_sum,
        'explanation_q_sum': explanation_q_sum,
        'answered_correctly_uq': answered_correctly_uq,
        'timestamp_u': timestamp_u,
        'timestamp_u_incorrect': timestamp_u_incorrect,
        'answered_correctly_up_count': answered_correctly_up_count,
        'answered_correctly_up_sum':answered_correctly_up_sum,
    }
    
    return prior_question_elapsed_time_mean, features_dicts


def inference(TARGET, FEATURES, sakt_model, lgb_model, prior_question_elapsed_time_mean, features_dicts):
    
    # Get feature dict
    answered_correctly_u_count = features_dicts['answered_correctly_u_count']
    answered_correctly_u_sum = features_dicts['answered_correctly_u_sum']
    elapsed_time_u_sum = features_dicts['elapsed_time_u_sum']
    explanation_u_sum = features_dicts['explanation_u_sum']
    answered_correctly_q_count = features_dicts['answered_correctly_q_count']
    answered_correctly_q_sum = features_dicts['answered_correctly_q_sum']
    answered_correctly_uq = features_dicts["answered_correctly_uq"]
    elapsed_time_q_sum = features_dicts['elapsed_time_q_sum']
    explanation_q_sum = features_dicts['explanation_q_sum']
    timestamp_u = features_dicts['timestamp_u']
    timestamp_u_incorrect = features_dicts['timestamp_u_incorrect']
    answered_correctly_up_count = features_dicts['answered_correctly_up_count']
    answered_correctly_up_sum = features_dicts['answered_correctly_up_sum']
    
    
    # Get api iterator and predictor
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict
    questions_df = pd.read_pickle(question_file)
    questions_df.part = questions_df.part.astype(np.int8)
    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
            update_features(previous_test_df, answered_correctly_u_sum,answered_correctly_u_count, answered_correctly_q_sum, answered_correctly_q_count,timestamp_u_incorrect, answered_correctly_uq, answered_correctly_up_count, answered_correctly_up_sum)
            previous_test_df = previous_test_df[previous_test_df.content_type_id == False]
            
            prev_group = previous_test_df[['user_id', 'content_id', 'answered_correctly','part', 'prior_question_elapsed_time_sakt']].groupby('user_id').apply(lambda r: (
                r['content_id'].values,
                r['answered_correctly'].values,
                r['part'].values,
                r['prior_question_elapsed_time_sakt'].values
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

        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time_sakt'] = test_df['prior_question_elapsed_time']/3600
        test_df.prior_question_elapsed_time_sakt.fillna(prior_question_elapsed_time_mean_sakt, inplace=True)
        test_df.prior_question_elapsed_time_sakt.clip(lower=0,upper=16,inplace=True)
        test_df['prior_question_elapsed_time_sakt'] = test_df['prior_question_elapsed_time_sakt'].astype(np.int16)
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    
        test_df = pd.merge(test_df, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
        previous_test_df = test_df.copy()
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        test_dataset = TestDataset(group, test_df, skills)
        test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)

        outs = []

        for item in test_dataloader:
            x = item[0].to(device).long()
            p = item[1].to(device).long()
            e_time = item[2].to(device).long()
            target_id = item[3].to(device).long()

            with torch.no_grad():
                output, att_weight = model(x, p, e_time, target_id)
            outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())
        test_df[TARGET] = 0
        test_df = add_features(test_df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, timestamp_u, timestamp_u_incorrect, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, answered_correctly_up_count, answered_correctly_up_sum, update=False)
        test_df[TARGET] =  lgb_model.predict(test_df[FEATURES]) * 0.6 + np.array(outs) * 0.4
        set_predict(test_df[['row_id', TARGET]])
        
    print('Job Done')
    

    
    
    
    
prior_question_elapsed_time, features_dicts = read_and_preprocess(feature_engineering=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)
try:
    model.load_state_dict(torch.load("../input/exp-040/sakt_model.pt"))
except:
    model.load_state_dict(torch.load("../input/exp-040/sakt_model.pt", map_location='cpu'))
model.to(device)
model.eval()

group = joblib.load("../input/exp-040/group.pkl.zip")
# questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
# questions_df.part = questions_df.part.astype(np.int8)
prior_question_elapsed_time_mean_sakt = 7.0621662139


lgb_model = lgb.Booster(model_file='../input/exp-013/lgb_exp_13.txt')
print('inference started...')
inference(TARGET, FEATURES, model, lgb_model, prior_question_elapsed_time, features_dicts)