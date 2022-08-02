# import argparse
# from AFSD.evaluation.eval_detection import ANETdetection

# parser = argparse.ArgumentParser()
# parser.add_argument('output_json', type=str)
# parser.add_argument('gt_json', type=str, default='./thumos_annotations/thumos_gt.json', nargs='?')
# args = parser.parse_args()

# tious = [0.3, 0.4, 0.5, 0.6, 0.7]
# anet_detection = ANETdetection(
#     ground_truth_filename=args.gt_json,
#     prediction_filename=args.output_json,
#     subset='test', tiou_thresholds=tious)
# mAPs, average_mAP, ap = anet_detection.evaluate()
# for (tiou, mAP) in zip(tious, mAPs):
#     print("mAP at tIoU {} is {}".format(tiou, mAP))



from evaluation.eval_detection import ANETdetection

print("Evaluation Started")
anet_detection = ANETdetection(
    ground_truth_filename="./evaluation/activity_net_1_3_new.json",
    # prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
    prediction_filename="output_gsm.json",
    subset='validation', verbose=False, check_status=False)
anet_detection.evaluate()
print("Evaluation Finished")
mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
print(results)
# with open(os.path.join(opt['output'], 'results.txt'), 'a') as fobj:
#     fobj.write(f'{results}\n')