import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
import riiideducation
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

import random
import os
import warnings
warnings.simplefilter('ignore')

#=====================================================================
# Riiidコンペで使用したbest lgb modelのtrainingファイル



# Random seed
SEED = 123

DEBUG = False


ver = 13

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(SEED)


def make_question_stats(df, tr):
    content_agg = tr.groupby('content_id')['answered_correctly'].agg(['mean','std'])
    df['answered_correctly_q_mean'] = df['question_id'].map(content_agg['mean']).astype(np.float16)
    df['answered_correctly_q_std'] = df['question_id'].map(content_agg['std']).astype(np.float16)
    del content_agg;gc.collect()
    part_agg = df.groupby('part')['answered_correctly_q_mean'].agg(['mean','std']).fillna(0)
    df['answered_correctly_p_mean'] = df['part'].map(part_agg['mean']).astype(np.float16)
    df['answered_correctly_p_std'] = df['part'].map(part_agg['std']).astype(np.float16)
    del part_agg;gc.collect()
    bundle_agg = df.groupby('bundle_id')['answered_correctly_q_mean'].agg(['mean','std']).fillna(0)
    df['answered_correctly_b_mean'] = df['bundle_id'].map(bundle_agg['mean']).astype(np.float16)
    df['answered_correctly_b_std'] = df['bundle_id'].map(bundle_agg['std']).astype(np.float16)
    del bundle_agg;gc.collect()
    df['tags'] = df['tags'].fillna(0)
    df['tag_1'] = df['tags'].str.split(' ',n=10,expand=True)[0]
    df['tag_1'] = df['tag_1'].fillna(0).astype(np.int32)
    tag_1_agg = df.groupby('tag_1')['answered_correctly_q_mean'].agg(['mean','std'])
    df['answered_correctly_tag_1_std'] = df['tag_1'].map(tag_1_agg['std']).astype(np.float16)
    df['answered_correctly_tag_1_mean'] = df['tag_1'].map(tag_1_agg['mean']).astype(np.float16)
    del tag_1_agg;gc.collect()
    return df



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
#             timestamp_u_recency_4[num] = np.nan
#             timestamp_u_recency_5[num] = np.nan
        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
#             timestamp_u_recency_4[num] = np.nan
#             timestamp_u_recency_5[num] = np.nan
        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan
#             timestamp_u_recency_4[num] = np.nan
#             timestamp_u_recency_5[num] = np.nan
        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]
#             timestamp_u_recency_4[num] = np.nan
#             timestamp_u_recency_5[num] = np.nan
#         elif len(timestamp_u[row[0]]) == 4:
#             timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][3]
#             timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][2]
#             timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][1]
#             timestamp_u_recency_4[num] = row[5] - timestamp_u[row[0]][0]
#             timestamp_u_recency_5[num] = np.nan
#         elif len(timestamp_u[row[0]]) == 5:
#             timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][4]
#             timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][3]
#             timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][2]
#             timestamp_u_recency_4[num] = row[5] - timestamp_u[row[0]][1]
#             timestamp_u_recency_5[num] = row[5] - timestamp_u[row[0]][0]   
        
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
    question_file = '../input/riiid-test-answer-prediction/questions.csv'
    
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
    questions_df = pd.read_csv(question_file)
    questions_df['part'] = questions_df['part'].astype(np.int32)
    questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)
    questions_df = make_question_stats(questions_df, train)
    question_cols = ['question_id','part','tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']
    train = pd.merge(train, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    valid = pd.merge(valid, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    questions_df.to_pickle('questions_df.pkl')
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
    train = add_features(train, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u,timestamp_u_incorrect, 
                         answered_correctly_q_count, answered_correctly_q_sum, 
                         elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq,
                         answered_correctly_up_count, answered_correctly_up_sum, update=True
                         )
    valid = add_features(valid, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u, timestamp_u_incorrect, 
                         answered_correctly_q_count, answered_correctly_q_sum, 
                         elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq,
                         answered_correctly_up_count, answered_correctly_up_sum, update=True
                         )
    
    gc.collect()
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
    
    return train, valid, prior_question_elapsed_time_mean, features_dicts

# Function for training and evaluation
def train_and_evaluate(train, valid, feature_engineering = False):
    
    TARGET = 'answered_correctly'
    # Features to train and predict
    FEATURES = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'part', 'answered_correctly_u_avg','answered_correctly_h_avg', 'elapsed_time_u_avg', 'explanation_u_avg',
                'answered_correctly_q_avg', 'elapsed_time_q_avg', 'explanation_q_avg', 'answered_correctly_uq_count', 
                'timestamp_u_recency_1', 'timestamp_u_recency_2', 'timestamp_u_recency_3', 
                'timestamp_u_incorrect_recency', 'answered_correctly_up_avg',
                'tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']
                #'timestamp_u_recency_4', 'timestamp_u_recency_5',
    # Delete some training data to experiment faster
    if feature_engineering:
        if DEBUG:
            train = train.sample(20000, random_state = SEED)
        else:
            train = train.sample(17000000, random_state = SEED)
    gc.collect()
    print(f'Traning with {train.shape[0]} rows and {len(FEATURES)} features')    
    drop_cols = list(set(train.columns) - set(FEATURES))
    y_train = train[TARGET]
    y_val = valid[TARGET]
    # Drop unnecessary columns
    train.drop(drop_cols, axis = 1, inplace = True)
    valid.drop(drop_cols, axis = 1, inplace = True)
    gc.collect()
    
    lgb_train = lgb.Dataset(train[FEATURES], y_train)
    lgb_valid = lgb.Dataset(valid[FEATURES], y_val)
    del train, y_train
    gc.collect()
    
    params = {'objective': 'binary', 
              'seed': SEED,
              'metric': 'auc',
              'num_leaves': 200,
              'feature_fraction': 0.75,
              'bagging_freq': 10,
              'bagging_fraction': 0.80
             }
    
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 10000,
        valid_sets = [lgb_train, lgb_valid],
        early_stopping_rounds = 10,
        verbose_eval = 50
    )
    
    print('Our Roc Auc score for the validation data is:', roc_auc_score(y_val, model.predict(valid[FEATURES])))
    
    feature_importance = model.feature_importance()
    feature_importance = pd.DataFrame({'Features': FEATURES, 'Importance': feature_importance}).sort_values('Importance', ascending = False)
    
    fig = plt.figure(figsize = (10, 10))
    fig.suptitle('Feature Importance', fontsize = 20)
    plt.tick_params(axis = 'x', labelsize = 12)
    plt.tick_params(axis = 'y', labelsize = 12)
    plt.xlabel('Importance', fontsize = 15)
    plt.ylabel('Features', fontsize = 15)
    sns.barplot(x = feature_importance['Importance'], y = feature_importance['Features'], orient = 'h')
    plt.show()
    
    model.save_model(f'lgb_exp_{ver}.txt')
    
    return TARGET, FEATURES, model

# Using time series api that simulates production predictions
def inference(TARGET, FEATURES, model, prior_question_elapsed_time_mean, features_dicts):
    
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
    questions_df = pd.read_pickle('questions_df.pkl')
    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
            update_features(previous_test_df, answered_correctly_u_sum,answered_correctly_u_count, answered_correctly_q_sum, answered_correctly_q_count,timestamp_u_incorrect, answered_correctly_uq, answered_correctly_up_count, answered_correctly_up_sum)
        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
        question_cols = ['question_id','part','tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']
        test_df = pd.merge(test_df, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
        previous_test_df = test_df.copy()
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        test_df[TARGET] = 0
        test_df = add_features(test_df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, timestamp_u, timestamp_u_incorrect, answered_correctly_q_count, answered_correctly_q_sum, elapsed_time_q_sum, explanation_q_sum, answered_correctly_uq, answered_correctly_up_count, answered_correctly_up_sum, update=False)
        test_df[TARGET] =  model.predict(test_df[FEATURES])
        set_predict(test_df[['row_id', TARGET]])
        
    print('Job Done')
    
train, valid, prior_question_elapsed_time_mean, features_dicts = read_and_preprocess(feature_engineering = True)
TARGET, FEATURES, model = train_and_evaluate(train, valid, feature_engineering = True)

inference(TARGET, FEATURES, model, prior_question_elapsed_time_mean, features_dicts)