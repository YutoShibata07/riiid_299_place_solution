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
# Riiidコンペで使用したlecture-based lgbm のtraining ファイル


# Random seed
SEED = 123

DEBUG = False


ver = 28

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
                 answered_correctly_q_count, elapsed_time_q_sum, explanation_q_sum, answered_correctly_up_count,answered_correctly_up_sum, 
                 answered_correctly_tag1_count, answered_correctly_tag1_sum,update = True):
    # -----------------------------------------------------------------------
    # Client features
    answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
    explanation_u_avg = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)
    timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)

    
    
    timestamp_u_incorrect_recency = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Question features
    elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
    explanation_q_avg = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Client Part feature
    answered_correctly_up_avg = np.zeros(len(df), dtype=np.float32)
    # Client tag1 feature
    answered_correctly_tag1_avg = np.zeros(len(df), dtype=np.float32)
    
    
    
    for num, row in enumerate(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','part','tag_1']].values):
        
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
            elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        else:
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Client Part feature assignation
        if answered_correctly_up_count[row[0]][row[6]] != 0:
            answered_correctly_up_avg[num] = answered_correctly_up_sum[row[0]][row[6]]/answered_correctly_up_count[row[0]][row[6]]
        else:
            answered_correctly_up_avg[num] = np.nan
        # ------------------------------------------------------------------
        # Client tag1 feature assignation
        if answered_correctly_tag1_count[row[0]][row[-1]] != 0:
            answered_correctly_tag1_avg[num] = answered_correctly_tag1_sum[row[0]][row[-1]]/answered_correctly_tag1_count[row[0]][row[-1]]
        else:
            answered_correctly_tag1_avg[num] = np.nan
        
        
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
            # Client Part feature update
            answered_correctly_up_count[row[0]][row[6]] += 1
            answered_correctly_up_sum[row[0]][row[6]] += row[1]
            # Client tag1 feature update
            answered_correctly_tag1_count[row[0]][row[-1]] += 1
            answered_correctly_tag1_sum[row[0]][row[-1]] += row[1]
            
             
            
    user_df = pd.DataFrame({'answered_correctly_u_avg': answered_correctly_u_avg, 'elapsed_time_u_avg': elapsed_time_u_avg, 'explanation_u_avg': explanation_u_avg, 
                            'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg,
                            'answered_correctly_up_avg':answered_correctly_up_avg,
                            'answered_correctly_tag1_avg':answered_correctly_tag1_avg,
                            'timestamp_u_recency_1': timestamp_u_recency_1, 'timestamp_u_recency_2': timestamp_u_recency_2,
                            'timestamp_u_recency_3': timestamp_u_recency_3, 'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency,
                            })#'timestamp_u_recency_4': timestamp_u_recency_4,'timestamp_u_recency_5': timestamp_u_recency_5,  
    
    df = pd.concat([df, user_df], axis = 1)
    return df




def add_lectures_feats(df, curr_dict, tag_list_dict,lectures_df, q_taglist_df):
    new_df = df[["row_id", "user_id", "timestamp", "content_id", "content_type_id"]]
    new_df = new_df.merge(lectures_df, how="left", left_on = ["content_id","content_type_id"], right_on = ["lecture_id","content_type_id"])
    new_df = new_df.merge(q_taglist_df, how="left", left_on = ["content_id","content_type_id"], right_on = [q_taglist_df.index,"content_type_id"])
    new_df = new_df.sort_values(["timestamp"])
    new_df = new_df[['timestamp', 'user_id', 'content_type_id','tag','part_1','part_2','part_3','part_4','part_5','part_6','part_7',
                     'type_of_concept','type_of_intention','type_of_solving_question','type_of_starter','tags_l','row_id']]
    ulc_lb = np.zeros(len(df), dtype="int8")
    part1_l = np.zeros(len(df), dtype="uint16")
    part2_l = np.zeros(len(df), dtype="uint16")
    part3_l = np.zeros(len(df), dtype="uint16")
    part4_l = np.zeros(len(df), dtype="uint16")
    part5_l = np.zeros(len(df), dtype="uint16")
    part6_l = np.zeros(len(df), dtype="uint16")
    part7_l = np.zeros(len(df), dtype="uint16")
    type_of_concept_l = np.zeros(len(df), dtype="uint16")
    type_of_intention_l = np.zeros(len(df), dtype="uint16")
    type_of_solving_question_l = np.zeros(len(df), dtype="uint16")
    type_of_starter_l = np.zeros(len(df), dtype="uint16")
    has_tags_l = np.zeros(len(df), dtype="float32")
    
    # 0.'timestamp', 1.'user_id', 2.'content_type_id',3.'tag',4.'part_1',5.'part_2',6.'part_3',7.'part_4',8.'part_5',9.'part_6',10.'part_7',
    # 11.'type_of_concept',12.'type_of_intention',13.'type_of_solving_question',14.'type_of_starter',15.'tags_l', 16.'row_id'
    for cnt,row in enumerate(new_df.itertuples(index=False)):
        if int(row[2]) == 1:
            if row[1] in curr_dict:
                if row[2] == 1:
                    curr_dict[row[1]]["lecture_bool"] = 1
                    curr_dict[row[1]]["part_1_cnt"] += int(row[4])
                    curr_dict[row[1]]["part_2_cnt"] += int(row[5])
                    curr_dict[row[1]]["part_3_cnt"] += int(row[6])
                    curr_dict[row[1]]["part_4_cnt"] += int(row[7])
                    curr_dict[row[1]]["part_5_cnt"] += int(row[8])
                    curr_dict[row[1]]["part_6_cnt"] += int(row[9])
                    curr_dict[row[1]]["part_7_cnt"] += int(row[10])
                    curr_dict[row[1]]["type_of_concept_cnt"] += int(row[11])
                    curr_dict[row[1]]["type_of_intention_cnt"] += int(row[12])
                    curr_dict[row[1]]["type_of_solving_question_cnt"] += int(row[13])
                    curr_dict[row[1]]["type_of_starter_cnt"] += int(row[14])
                    tag_list_dict[row[1]].add(int(row[3]))#これまでに受けてきた授業のtag一覧
            else:
                curr_dict[row[1]] = {}
                if row[2] == 1:
                    curr_dict[row[1]]["lecture_bool"] = 1
                    curr_dict[row[1]]["part_1_cnt"] = int(row[4])
                    curr_dict[row[1]]["part_2_cnt"] = int(row[5])
                    curr_dict[row[1]]["part_3_cnt"] = int(row[6])
                    curr_dict[row[1]]["part_4_cnt"] = int(row[7])
                    curr_dict[row[1]]["part_5_cnt"] = int(row[8])
                    curr_dict[row[1]]["part_6_cnt"] = int(row[9])
                    curr_dict[row[1]]["part_7_cnt"] = int(row[10])
                    curr_dict[row[1]]["type_of_concept_cnt"] = int(row[11])
                    curr_dict[row[1]]["type_of_intention_cnt"] = int(row[12])
                    curr_dict[row[1]]["type_of_solving_question_cnt"] = int(row[13])
                    curr_dict[row[1]]["type_of_starter_cnt"] = int(row[14])
                    tag_list_dict[row[1]] = set([int(row[3])])
                else:
                    curr_dict[row[1]]["lecture_bool"] = 0
                    curr_dict[row[1]]["part_1_cnt"] = 0
                    curr_dict[row[1]]["part_2_cnt"] = 0
                    curr_dict[row[1]]["part_3_cnt"] = 0
                    curr_dict[row[1]]["part_4_cnt"] = 0
                    curr_dict[row[1]]["part_5_cnt"] = 0
                    curr_dict[row[1]]["part_6_cnt"] = 0
                    curr_dict[row[1]]["part_7_cnt"] = 0
                    curr_dict[row[1]]["type_of_concept_cnt"] = 0
                    curr_dict[row[1]]["type_of_intention_cnt"] = 0
                    curr_dict[row[1]]["type_of_solving_question_cnt"] = 0
                    curr_dict[row[1]]["type_of_starter_cnt"] = 0
                    tag_list_dict[row[1]] = set()
        
        ulc_lb[cnt] = curr_dict[row[1]]["lecture_bool"]
        part1_l[cnt] = curr_dict[row[1]]["part_1_cnt"]
        part2_l[cnt] = curr_dict[row[1]]["part_2_cnt"]
        part3_l[cnt] = curr_dict[row[1]]["part_3_cnt"]
        part4_l[cnt] = curr_dict[row[1]]["part_4_cnt"]
        part5_l[cnt] = curr_dict[row[1]]["part_5_cnt"]
        part6_l[cnt] = curr_dict[row[1]]["part_6_cnt"]
        part7_l[cnt] = curr_dict[row[1]]["part_7_cnt"]
        type_of_concept_l[cnt] = curr_dict[row[1]]["type_of_concept_cnt"]
        type_of_intention_l[cnt] = curr_dict[row[1]]["type_of_intention_cnt"]
        type_of_solving_question_l[cnt] = curr_dict[row[1]]["type_of_solving_question_cnt"]
        type_of_starter_l[cnt] = curr_dict[row[1]]["type_of_starter_cnt"]
        
        if type(row[15]) == list:#question_colにのみ行う
            tags_has = 0
            for tag in row[15]:
                if int(tag) in tag_list_dict[row[1]]:
                    tags_has += 1
            has_tags_l[cnt] = tags_has/len(row[15])#問題のtagの中で既に授業を受けたことがあるtagの割合。高ければ対策済みということ

    has_tags_lb = (has_tags_l > 0).astype("int8")

    lectures_feats_df = pd.DataFrame({"curr_lecture_bool":ulc_lb,
                                      "part_1_cnt":part1_l,
                                      "part_2_cnt":part2_l,
                                      "part_3_cnt":part3_l,
                                      "part_4_cnt":part4_l,
                                      "part_5_cnt":part5_l,
                                      "part_6_cnt":part6_l,
                                      "part_7_cnt":part7_l,
                                      "type_of_concept_cnt":type_of_concept_l,
                                      "type_of_intention_cnt":type_of_intention_l,
                                      "type_of_solving_question_cnt":type_of_solving_question_l,
                                      "type_of_starter_cnt":type_of_starter_l,
                                      "watched_tags_rate":has_tags_l,
                                      "watched_tags_bool":has_tags_lb,
                                     }).set_index(new_df["row_id"])

    df = df.merge(lectures_feats_df,how="left",left_on="row_id",right_index=True)
    return df





def update_features(df, answered_correctly_u_sum,answered_correctly_u_count, answered_correctly_q_count,timestamp_u_incorrect, answered_correctly_up_count, answered_correctly_up_sum, answered_correctly_tag1_count, answered_correctly_tag1_sum):
    for row in df[['user_id', 'answered_correctly', 'content_id', 'content_type_id', 'timestamp','part','tag_1']].values:
        if row[3] == 0:
            # ------------------------------------------------------------------
            # Count feature updates
            answered_correctly_u_count[row[0]] += 1
            answered_correctly_q_count[row[2]] += 1
            
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[4])
                else:
                    timestamp_u_incorrect[row[0]].append(row[4])
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------    
            # Client Part feature update
            answered_correctly_up_count[row[0]][row[5]] += 1
            answered_correctly_up_sum[row[0]][row[5]] += row[1]
            # Client tag1 feature update
            answered_correctly_tag1_count[row[0]][row[-1]] += 1
            answered_correctly_tag1_sum[row[0]][row[-1]] += row[1]
            
    return
    

def read_and_preprocess(lectures_df, q_taglist_df, feature_engineering = False):
    
    train_pickle = '../input/riiid-cross-validation-files/cv1_train.pickle'
    valid_pickle = '../input/riiid-cross-validation-files/cv1_valid.pickle'
    question_file = '../input/riiid-test-answer-prediction/questions.csv'
    
    # Read data
    feld_needed = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation','row_id']
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
            train = train.iloc[-10000000:]
            
#     Merge with question dataframe
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
    elapsed_time_q_sum = defaultdict(int)
    explanation_q_sum = defaultdict(int)
    
    # Client Part dictionary 
    answered_correctly_up_count = defaultdict(lambda:defaultdict(int))
    answered_correctly_up_sum = defaultdict(lambda:defaultdict(int))
    # Client Tag1 dictionary
    answered_correctly_tag1_count = defaultdict(lambda:defaultdict(int))
    answered_correctly_tag1_sum = defaultdict(lambda:defaultdict(int))
    
    lect_dict = defaultdict(lambda:defaultdict(int))
    tag_list_dict = defaultdict(set)
    
    
    
    train = add_lectures_feats(train,lect_dict, tag_list_dict,lectures_df, q_taglist_df)
    valid = add_lectures_feats(valid, lect_dict, tag_list_dict,lectures_df, q_taglist_df)
    
    # Filter by content_type_id to discard lectures
    train = train.loc[train.content_type_id == False].reset_index(drop = True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)
    
    gc.collect()
    
    # Changing dtype to avoid lightgbm error
    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
    valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')
    
    # Fill prior question elapsed time with the mean
    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    

    
    
    print('User feature calculation started...')
    train = add_features(train, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u,timestamp_u_incorrect, 
                         answered_correctly_q_count,
                         elapsed_time_q_sum, explanation_q_sum,
                         answered_correctly_up_count, answered_correctly_up_sum, 
                         answered_correctly_tag1_count, answered_correctly_tag1_sum,update=True
                         )
    valid = add_features(valid, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, 
                         timestamp_u, timestamp_u_incorrect, 
                         answered_correctly_q_count, 
                         elapsed_time_q_sum, explanation_q_sum, 
                         answered_correctly_up_count, answered_correctly_up_sum, 
                         answered_correctly_tag1_count,answered_correctly_tag1_sum,update=True
                         )
    
    gc.collect()
    print('User feature calculation completed...')
    
    features_dicts = {
        'answered_correctly_u_count': answered_correctly_u_count,
        'answered_correctly_u_sum': answered_correctly_u_sum,
        'elapsed_time_u_sum': elapsed_time_u_sum,
        'explanation_u_sum': explanation_u_sum,
        'answered_correctly_q_count': answered_correctly_q_count,
        'elapsed_time_q_sum': elapsed_time_q_sum,
        'explanation_q_sum': explanation_q_sum,
        'timestamp_u': timestamp_u,
        'timestamp_u_incorrect': timestamp_u_incorrect,
        'answered_correctly_up_count': answered_correctly_up_count,
        'answered_correctly_up_sum':answered_correctly_up_sum,
        'answered_correctly_tag1_count': answered_correctly_tag1_count,
        'answered_correctly_tag1_sum':answered_correctly_tag1_sum,
        'lect_dict':lect_dict,
        'tag_list_dict':tag_list_dict
    }
    
    return train, valid, prior_question_elapsed_time_mean, features_dicts

# Function for training and evaluation
def train_and_evaluate(train, valid, feature_engineering = False):
    
    TARGET = 'answered_correctly'
    # Features to train and predict
    FEATURES = ['prior_question_elapsed_time', 'prior_question_had_explanation', 'part', 'answered_correctly_u_avg','elapsed_time_u_avg', 'explanation_u_avg',
                'elapsed_time_q_avg', 'explanation_q_avg', 
                'timestamp_u_recency_1', 'timestamp_u_recency_2', 'timestamp_u_recency_3', 
                'timestamp_u_incorrect_recency', 'answered_correctly_up_avg','answered_correctly_tag1_avg',
                'tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std',
                'curr_lecture_bool','part_1_cnt','part_2_cnt','part_3_cnt','part_5_cnt','part_6_cnt','part_7_cnt',
                'type_of_concept_cnt','type_of_intention_cnt','type_of_solving_question_cnt','type_of_starter_cnt', "watched_tags_rate", "watched_tags_bool"]
                
    # Delete some training data to experiment faster
    if feature_engineering:
        if DEBUG:
            train = train.sample(20000, random_state = SEED)
#         else:
#             train = train.sample(17000000, random_state = SEED)
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
def inference(TARGET, FEATURES, model, prior_question_elapsed_time_mean, features_dicts, lectures_df, q_taglist_df):
    
    # Get feature dict
    answered_correctly_u_count = features_dicts['answered_correctly_u_count']
    answered_correctly_u_sum = features_dicts['answered_correctly_u_sum']
    elapsed_time_u_sum = features_dicts['elapsed_time_u_sum']
    explanation_u_sum = features_dicts['explanation_u_sum']
    answered_correctly_q_count = features_dicts['answered_correctly_q_count']

    elapsed_time_q_sum = features_dicts['elapsed_time_q_sum']
    explanation_q_sum = features_dicts['explanation_q_sum']
    timestamp_u = features_dicts['timestamp_u']
    timestamp_u_incorrect = features_dicts['timestamp_u_incorrect']
    answered_correctly_up_count = features_dicts['answered_correctly_up_count']
    answered_correctly_up_sum = features_dicts['answered_correctly_up_sum']
    answered_correctly_tag1_count = features_dicts['answered_correctly_tag1_count']
    answered_correctly_tag1_sum = features_dicts['answered_correctly_tag1_sum']
    
    lect_dict = features_dicts['lect_dict']
    tag_list_dict = features_dicts['tag_list_dict']
    
    # Get api iterator and predictor
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict
    questions_df = pd.read_pickle('questions_df.pkl')
    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
            update_features(previous_test_df, answered_correctly_u_sum,answered_correctly_u_count, answered_correctly_q_count,timestamp_u_incorrect, answered_correctly_up_count, answered_correctly_up_sum, answered_correctly_tag1_count, answered_correctly_tag1_sum)
        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
        question_cols = ['question_id','part','tag_1','answered_correctly_q_mean','answered_correctly_q_std','answered_correctly_p_mean','answered_correctly_p_std','answered_correctly_b_mean','answered_correctly_b_std','answered_correctly_tag_1_mean','answered_correctly_tag_1_std']
        test_df = pd.merge(test_df, questions_df[question_cols], left_on = 'content_id', right_on = 'question_id', how = 'left')
        previous_test_df = test_df.copy()
        test_df = add_lectures_feats(test_df, lect_dict, tag_list_dict, lectures_df, q_taglist_df)
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        test_df[TARGET] = 0
        test_df = add_features(test_df, answered_correctly_u_count, answered_correctly_u_sum, elapsed_time_u_sum, explanation_u_sum, timestamp_u, timestamp_u_incorrect, answered_correctly_q_count, elapsed_time_q_sum, explanation_q_sum, answered_correctly_up_count, answered_correctly_up_sum, answered_correctly_tag1_count, answered_correctly_tag1_sum,update=False)
        test_df[TARGET] =  model.predict(test_df[FEATURES])
        set_predict(test_df[['row_id', TARGET]])
        
    print('Job Done')
    
    
lectures_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')
lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')
lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])
lectures_df['content_type_id'] = 1
q_taglist_df = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/questions.csv")[['tags']].astype(str)
q_taglist_df["tags_l"] = [x.split() for x in q_taglist_df.tags.values]
q_taglist_df['content_type_id'] = 0
q_taglist_df.drop("tags", axis=1, inplace=True)
q_taglist_df.drop(10033, axis=0, inplace=True) # nan


    
train, valid, prior_question_elapsed_time_mean, features_dicts = read_and_preprocess(lectures_df, q_taglist_df,feature_engineering = True)

TARGET, FEATURES, model = train_and_evaluate(train, valid, feature_engineering = True)

inference(TARGET, FEATURES, model, prior_question_elapsed_time_mean, features_dicts, lectures_df, q_taglist_df)