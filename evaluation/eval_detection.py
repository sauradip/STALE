

# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code

import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou
from config.dataset_class import activity_dict

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

pred_data = pd.read_csv("stale_output.csv")
pred_videos = set(list(pred_data.video_name.values[:]))
# print(len(pred_videos))

class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        # activity_index = {'Beer pong': 0, 'Kneeling': 1, 'Tumbling': 2, 'Sharpening knives': 3, 'Playing water polo': 4, 'Scuba diving': 5, 'Arm wrestling': 6, 'Archery': 7, 'Shaving': 8, 'Playing bagpipes': 9, 'Riding bumper cars': 10, 'Surfing': 11, 'Hopscotch': 12, 'Gargling mouthwash': 13, 'Playing violin': 14, 'Plastering': 15, 'Changing car wheel': 16, 'Horseback riding': 17, 'Playing congas': 18, 'Doing a powerbomb': 19, 'Walking the dog': 20, 'Using the pommel horse': 21, 'Rafting': 22, 'Hurling': 23, 'Removing curlers': 24, 'Windsurfing': 25, 'Playing drums': 26, 'Tug of war': 27, 'Playing badminton': 28, 'Getting a piercing': 29, 'Camel ride': 30, 'Sailing': 31, 'Wrapping presents': 32, 'Hand washing clothes': 33, 'Braiding hair': 34, 'Using the monkey bar': 35, 'Longboarding': 36, 'Doing motocross': 37, 'Cleaning shoes': 38, 'Vacuuming floor': 39, 'Blow-drying hair': 40, 'Doing fencing': 41, 'Playing harmonica': 42, 'Playing blackjack': 43, 'Discus throw': 44, 'Playing flauta': 45, 'Ice fishing': 46, 'Spread mulch': 47, 'Mowing the lawn': 48, 'Capoeira': 49, 'Preparing salad': 50, 'Beach soccer': 51, 'BMX': 52, 'Playing kickball': 53, 'Shoveling snow': 54, 'Swimming': 55, 'Cheerleading': 56, 'Removing ice from car': 57, 'Calf roping': 58, 'Breakdancing': 59, 'Mooping floor': 60, 'Powerbocking': 61, 'Kite flying': 62, 'Running a marathon': 63, 'Swinging at the playground': 64, 'Shaving legs': 65, 'Starting a campfire': 66, 'River tubing': 67, 'Zumba': 68, 'Putting on makeup': 69, 'Raking leaves': 70, 'Canoeing': 71, 'High jump': 72, 'Futsal': 73, 'Hitting a pinata': 74, 'Wakeboarding': 75, 'Playing lacrosse': 76, 'Grooming dog': 77, 'Cricket': 78, 'Getting a tattoo': 79, 'Playing saxophone': 80, 'Long jump': 81, 'Paintball': 82, 'Tango': 83, 'Throwing darts': 84, 'Ping-pong': 85, 'Tennis serve with ball bouncing': 86, 'Triple jump': 87, 'Peeling potatoes': 88, 'Doing step aerobics': 89, 'Building sandcastles': 90, 'Elliptical trainer': 91, 'Baking cookies': 92, 'Rock-paper-scissors': 93, 'Playing piano': 94, 'Croquet': 95, 'Playing squash': 96, 'Playing ten pins': 97, 'Using parallel bars': 98, 'Snowboarding': 99, 'Preparing pasta': 100, 'Trimming branches or hedges': 101, 'Playing guitarra': 102, 'Cleaning windows': 103, 'Playing field hockey': 104, 'Skateboarding': 105, 'Rollerblading': 106, 'Polishing shoes': 107, 'Fun sliding down': 108, 'Smoking a cigarette': 109, 'Spinning': 110, 'Disc dog': 111, 'Installing carpet': 112, 'Using the balance beam': 113, 'Drum corps': 114, 'Playing polo': 115, 'Doing karate': 116, 'Hammer throw': 117, 'Baton twirling': 118, 'Tai chi': 119, 'Kayaking': 120, 'Grooming horse': 121, 'Washing face': 122, 'Bungee jumping': 123, 'Clipping cat claws': 124, 'Putting in contact lenses': 125, 'Playing ice hockey': 126, 'Brushing hair': 127, 'Welding': 128, 'Mixing drinks': 129, 'Smoking hookah': 130, 'Having an ice cream': 131, 'Chopping wood': 132, 'Plataform diving': 133, 'Dodgeball': 134, 'Clean and jerk': 135, 'Snow tubing': 136, 'Decorating the Christmas tree': 137, 'Rope skipping': 138, 'Hand car wash': 139, 'Doing kickboxing': 140, 'Fixing the roof': 141, 'Playing pool': 142, 'Assembling bicycle': 143, 'Making a sandwich': 144, 'Shuffleboard': 145, 'Curling': 146, 'Brushing teeth': 147, 'Fixing bicycle': 148, 'Javelin throw': 149, 'Pole vault': 150, 'Playing accordion': 151, 'Bathing dog': 152, 'Washing dishes': 153, 'Skiing': 154, 'Playing racquetball': 155, 'Shot put': 156, 'Drinking coffee': 157, 'Hanging wallpaper': 158, 'Layup drill in basketball': 159, 'Springboard diving': 160, 'Volleyball': 161, 'Ballet': 162, 'Rock climbing': 163, 'Ironing clothes': 164, 'Snatch': 165, 'Drinking beer': 166, 'Roof shingle removal': 167, 'Blowing leaves': 168, 'Cumbia': 169, 'Hula hoop': 170, 'Waterskiing': 171, 'Carving jack-o-lanterns': 172, 'Cutting the grass': 173, 'Sumo': 174, 'Making a cake': 175, 'Painting fence': 176, 'Doing crunches': 177, 'Making a lemonade': 178, 'Applying sunscreen': 179, 'Painting furniture': 180, 'Washing hands': 181, 'Painting': 182, 'Putting on shoes': 183, 'Knitting': 184, 'Doing nails': 185, 'Getting a haircut': 186, 'Using the rowing machine': 187, 'Polishing forniture': 188, 'Using uneven bars': 189, 'Playing beach volleyball': 190, 'Cleaning sink': 191, 'Slacklining': 192, 'Bullfighting': 193, 'Table soccer': 194, 'Waxing skis': 195, 'Playing rubik cube': 196, 'Belly dance': 197, 'Making an omelette': 198, 'Laying tile': 199}
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            # print(videoid, pred_videos[2:])
            if "v_"+videoid in pred_videos:
                
                # print(v)
                if self.subset != v['subset']:
                    continue
                if videoid in self.blocked_videos:
                    continue
                for ann in v['annotations']:
                    if ann['label'] not in activity_index:
                        activity_index[ann['label']] = cidx
                        cidx += 1
                    video_lst.append(videoid)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        activity_index_prep = activity_dict
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                  label = activity_index_prep[result['label']]
                else:
                  label = self.activity_index[result['label']]
                # label = self.activity_index[result['label']]
                video_lst.append(videoid)
                # print("pred_vid",videoid)
                # print("pred_label",result['label'])
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])


    return ap