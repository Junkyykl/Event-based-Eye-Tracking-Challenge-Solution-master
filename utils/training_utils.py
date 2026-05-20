import torch
import os
from utils.metrics import p_acc, p_acc_wo_closed_eye, px_euclidean_dist, process_array
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.functional import binary_cross_entropy_with_logits
import pdb
import numpy as np
import imageio
import cv2
def visualize_training_samples(events, targets):
    events_ = events.detach().cpu().numpy()[2]
    targets_ = targets.detach().cpu().numpy()[2]
    events = events_[:, 0] - events_[:, 1]
    events = np.stack([events, events, events], axis=1)
    events = (events - np.min(events)) / (np.max(events) - np.min(events)) * 255
    events = events.astype(np.uint8)
    events = np.transpose(events, (0, 2, 3, 1))
    events_ = events.copy()
    for i in range(targets_.shape[0]):
        cv2.circle(events_[i], (int(targets_[i][0]) , int(targets_[i][1])), 8, (0, 255, 0), -1)
        cv2.circle(events_[i], (int(targets_[i][0]) , int(targets_[i][1])), 80, (0, 255, 0), 2)
    os.makedirs('./vis_sim_1', exist_ok=True)
    imageio.mimsave(f'./vis_sim_1/training.gif', events_, duration=1000/15)

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args):
    alpha = 0.0
    model.train()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_error_all  = 0, 0

    iters = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        # visualize_training_samples(inputs, targets)
        # pdb.set_trace()
        optimizer.zero_grad()
        outputs = model(inputs.to("cuda"))
        #taking only the last frame's label, and first two dim are coordinate, last is open or close so discarded
        targets = targets.to("cuda")
        loss = criterion(outputs, targets[:,:, :2]) 
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + i / iters) # for cosine annealing scheduler
            # lr_scheduler.step() # for step scheduler
        total_loss += loss.item()

        # calculate pixel tolerated accuracy
        p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor, \
                                    pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance
        p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :3], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor)
        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen
    
    metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return model, total_loss / len(train_loader), metrics


def validate_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}
    total_samples_all, total_sample_p_error_all  = 0, 0
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to("cuda"))
            targets = targets.to("cuda")

            loss = criterion(outputs, targets[:,:, :2]) 
 
            total_loss += loss.item()
            outputs_list.append(outputs.detach().cpu())
            targets_list.append(targets.detach().cpu())
    # pdb.set_trace()
    outputs_list = torch.cat(outputs_list, dim=0)
    targets_list = torch.cat(targets_list, dim=0)
    
    outputs_list = torch.cat([outputs_list, targets_list[:, :, -1].unsqueeze(-1)], dim=2)
    # pdb.set_trace()
    outputs, targets = process_array(outputs_list, targets_list[:, :, :3])
    outputs = np.concatenate(outputs, axis=0)
    # pdb.set_trace()
    targets = np.concatenate(targets, axis=0)
    outputs, targets = torch.tensor(outputs), torch.tensor(targets)
    p_corr, batch_size = p_acc(targets[:, :2], outputs[:, :2], \
                        width_scale=args.sensor_width*args.spatial_factor, \
                        height_scale=args.sensor_height*args.spatial_factor, \
                            pixel_tolerances=args.pixel_tolerances)
    total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
    total_samples_all += batch_size
    p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :3], outputs[:, :2], \
                        width_scale=args.sensor_width*args.spatial_factor, \
                        height_scale=args.sensor_height*args.spatial_factor)
    total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
    total_sample_p_error_all += bs_times_seqlen
    metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
                'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return total_loss / len(val_loader), metrics


def top_k_checkpoints(args):
    """
    only save the top k model checkpoints with the lowest validation loss.
    """
    # ...
    model_checkpoints = [f for f in os.listdir(args.log_dir) if f.endswith(".pth")]
    if len(model_checkpoints) > args.save_k_best:
        model_checkpoints = sorted([f for f in os.listdir(args.log_dir) if f.startswith("ep")], key=lambda x: float(x.split("_")[-2]))
        os.remove(os.path.join(args.log_dir, model_checkpoints[-1]))


