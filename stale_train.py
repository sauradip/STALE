import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch import autograd
import numpy as np
from stale_model import STALE
import yaml
import stale_lib.stale_dataloader as stale_dataset
from stale_lib.loss_stale import stale_loss
from config.dataset_class import activity_dict



with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

output_path=config['dataset']['training']['output_path']
num_gpu = config['training']['num_gpu']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
decay = config['training']['weight_decay']
epoch = config['training']['max_epoch']
num_batch = config['training']['batch_size']
step_train = config['training']['step']
gamma_train = config['training']['gamma']
fix_seed = config['training']['random_seed']
pretrain_mode = config['model']['clip_pretrain']
split = config['dataset']['split']



################## fix everything ##################
import random
seed = fix_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################

# reduce_feat = nn.Conv1d(2048, 1024, 1)

def get_mem_usage():
    GB = 1024.0 ** 3
    output = ["device_%d = %.03fGB" % (device, torch.cuda.max_memory_allocated(torch.device('cuda:%d' % device)) / GB) for device in range(num_gpu)]
    return ' '.join(output)[:-1]

# training-CLIP
def train(data_loader, model, optimizer, epoch,scheduler):
    model.train()
    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt, bot_gt) in enumerate(data_loader):
        top_br_pred, bottom_br_pred , mask_pred, cls_pred, features = model(input_data.cuda(),"train")
        loss = stale_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred,action_gt, mask_pred,bot_gt,cls_pred,label_gt,features,"train")
        tot_loss = loss[0]
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + M-Loss {4:.2f}  (train)".format(
    epoch, tot_loss,loss[1],loss[2],loss[3]))

# validation
def test(data_loader, model, epoch, best_loss):
    model.eval()
    with torch.no_grad():
      for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt,bot_gt) in enumerate(data_loader):
        top_br_pred, bottom_br_pred, mask_pred,cls_pred, features = model(input_data.cuda(),"test")
        loss = stale_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred,action_gt, mask_pred, bot_gt,cls_pred,label_gt,features, "test")
        tot_loss = loss[0]
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + M-Loss {4:.2f} (val)".format(
    epoch, tot_loss,loss[1],loss[2],loss[3]))
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, output_path + "/STALE_checkpoint_"+ str(split) + "_split.pth.tar")
    if tot_loss < best_loss:
        best_loss = tot_loss
        torch.save(state, output_path + "/STALE_best_"+ str(split) + "_split.pth.tar")

    return best_loss

if __name__ == '__main__':

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = STALE()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()
    for param in model.parameters():
        param.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal Number of Learnable Paramters (in M) : ",total_params/1000000)
    print('No of Gpus using to Train :  {} '.format(num_gpu))
    print(" Saving all Checkpoints in path : "+ output_path )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=decay)

    train_loader = torch.utils.data.DataLoader(stale_dataset.STALEDataset(subset="train"),
                                               batch_size=num_batch, shuffle=True,
                                               num_workers=8, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(stale_dataset.STALEDataset(subset="validation"),
                                              batch_size=num_batch, shuffle=False,
                                              num_workers=8, pin_memory=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_train, gamma=gamma_train)
    best_loss = 1e10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(epoch):
      with autograd.detect_anomaly():
        train(train_loader, model, optimizer, epoch,scheduler)
        best_loss = test(test_loader, model, epoch, best_loss)
        scheduler.step()
    # writer.flush()
    end.record()
    torch.cuda.synchronize()

    print("Total Time taken for Running "+str(epoch)+" epoch is :"+ str(start.elapsed_time(end)/1000) + " secs")  # milliseconds



