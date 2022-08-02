import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from config.dataset_class import activity_dict
from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test

# from gsm_lib import opts
import yaml


with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


vid_info = config['dataset']['training']['video_info_path']
vid_anno = config['dataset']['training']['video_anno_path']
vid_path = config['training']['feature_path']
nms_thresh = config['testing']['nms_thresh']
split = config['dataset']['split']
# if split == 50:
#     label_list = list(t2_dict_test.keys())
# else:
#     label_list = list(t1_dict_test.keys())

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict():
    df = pd.read_csv(vid_info)
    json_data = load_json(vid_anno)
    database = json_data
    video_dict = {}
    video_label_dict={}
    for i in range(len(df)):
        video_name = df.video.values[i]
        # if os.path.exists(os.path.join(vid_path+"/validation",video_name+".npy")):
        if os.path.exists(os.path.join(vid_path+"/",video_name+".npy")):
            video_info = database[video_name]
            video_new_info = {}
            video_new_info['duration_frame'] = video_info['duration_frame']
            video_new_info['duration_second'] = video_info['duration_second']
            video_new_info["feature_frame"] = video_info['duration_frame']
            video_subset = df.subset.values[i]
            video_anno = video_info['annotations']
            video_new_info['annotations'] = video_info['annotations']
            if len(video_anno) > 0:
                video_label = video_info['annotations'][0]['label']
                if video_subset == 'validation' :
                        video_dict[video_name] = video_new_info
                        video_label_dict[video_name] = video_label
    return video_dict , video_label_dict



def Soft_NMS(df, nms_threshold=1e-5, num_prop=100):
 
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tlabel = list(df.label.values[:])

    rstart = []
    rend = []
    rscore = []
    rlabel = []


    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * (np.exp(-np.square(tmp_iou)*10) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rlabel.append(tlabel[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['label'] = rlabel

    return newDf



def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - min(s1, s2)
    return float(Aand) / (Aor - Aand + (e2-s2))



def multithread_detection(video_name, video_cls, video_info, label_dict, pred_prop, best_cls, num_prop=200, topk = 2):
    
    old_df = pred_prop[pred_prop.video_name == "v_"+video_name]
    # print(df)
    best_score = best_cls["v_"+video_name]["score"]
    best_label = best_cls["v_"+video_name]["class"]
    
    df = pd.DataFrame()
    df['score'] = old_df.reg_score.values[:]*old_df.clr_score.values[:]
    df['label'] = old_df.label.values[:]
    df['xmin'] = old_df.xmin.values[:]
    df['xmax'] = old_df.xmax.values[:]

    

    if len(df) > 1:
        df = Soft_NMS(df, nms_thresh)
    df = df.sort_values(by="score", ascending=False)
    video_duration=float(video_info["feature_frame"])/video_info["duration_frame"]*video_info["duration_second"]
    proposal_list = []

    for j in range(min(100, len(df))):
        
        tmp_proposal = {}
        tmp_proposal["label"] = str(df.label.values[j])
        if float(df.score.values[j]) > best_score:
            tmp_proposal["label"] = str(df.label.values[j])
        else:
            # tmp_proposal["label"] = str(df.label.values[j])
            tmp_proposal["label"] = best_label

        tmp_proposal["score"] = float(df.score.values[j])
        # tmp_proposal["score"] = float(best_score)
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)

    # for j in range(min(100, len(df))):
        
    #     tmp_proposal = {}
    #     tmp_proposal["label"] = best_label
    #     # if float(df.score.values[j]) > best_score:
    #         # tmp_proposal["label"] = str(df.label.values[j])
    #     # else:
    #     #     tmp_proposal["label"] = str(df.label.values[j])
    #         # tmp_proposal["label"] = best_label

    #     tmp_proposal["score"] = float(df.score.values[j])
    #     tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
    #                             min(1, df.xmax.values[j]) * video_duration]
    #     proposal_list.append(tmp_proposal)

    return {video_name: proposal_list}






