import os
import pdb
import time
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.dataset import QuestionPairsDataset, collate_data
from utils.misc import set_random_seeds
from utils.arguments import get_args
from models.baselines import *

use_cuda = torch.cuda.is_available()

def evaluate_model(opts, model, loader, criterion):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for i, d in enumerate(loader):
            batch_size = d['s1'].shape[0]
            if use_cuda:
                d['s1'] = d['s1'].cuda()
                d['s2'] = d['s2'].cuda()
                d['len1'] = d['len1'].cuda()
                d['len2'] = d['len2'].cuda()
                d['label'] = d['label'].cuda()

            out = model(d['s1'], d['s2'], d['len1'], d['len2'])
            _, pred_labels = torch.max(F.softmax(out, dim=1), dim=1)
            acc = torch.eq(pred_labels, d['label']).sum().float() / batch_size
            loss = criterion(out, d['label'])

            val_loss += loss.data.cpu().item()
            val_acc += acc.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_acc = val_acc / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_acc, time_taken

def train(opts):

    train_dataset = QuestionPairsDataset(opts.data_dir, split='train', \
        glove_emb_file=opts.glove_emb_file, maxlen=opts.maxlen)
    valid_dataset = QuestionPairsDataset(opts.data_dir, split='val', \
        glove_emb_file=opts.glove_emb_file, maxlen=opts.maxlen)

    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    if opts.arch == 'lstm_concat':
        model = LSTMWithConcatBaseline(hidden_size=opts.hidden_size, num_layers=opts.num_layers, \
            bidirectional=opts.bidirectional, glove_emb_file=opts.glove_emb_file, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'lstm_dist_angle':
        model = LSTMWithDistAngleBaseline(hidden_size=opts.hidden_size, num_layers=opts.num_layers, \
            bidirectional=opts.bidirectional, glove_emb_file=opts.glove_emb_file, pretrained_emb=opts.pretrained_emb)
    else:
        raise NotImplementedError('unsupported model architecture')

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.wd)
    elif opts.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opts.lr, momentum=opts.momentum, weight_decay=opts.wd)
    else:
        raise NotImplementedError('Unknown optimizer type')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_gamma)
    criterion = nn.CrossEntropyLoss()
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # for logging
    n_iter = 0
    writer = SummaryWriter(log_dir=opts.log_dir)
    loss_log = {'train/loss' : 0.0, 'train/acc' : 0.0}
    time_start = time.time()
    num_batches = 0

    # for choosing the best model
    best_val_acc = 0.0

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        for i, d in enumerate(train_loader):
            batch_size = d['s1'].shape[0]
            if use_cuda:
                d['s1'] = d['s1'].cuda()
                d['s2'] = d['s2'].cuda()
                d['len1'] = d['len1'].cuda()
                d['len2'] = d['len2'].cuda()
                d['label'] = d['label'].cuda()

            out = model(d['s1'], d['s2'], d['len1'], d['len2'])
            _, pred_labels = torch.max(F.softmax(out, dim=1), dim=1)
            acc = torch.eq(pred_labels, d['label']).sum().float() / batch_size
            loss = criterion(out, d['label'])

            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            # log the losses
            n_iter += 1
            num_batches += 1
            loss_log['train/loss'] += loss.data.cpu().item()
            loss_log['train/acc'] += acc.data.cpu().item()

            if num_batches != 0 and n_iter % opts.log_iter == 0:
                time_end = time.time()
                time_taken = time_end - time_start
                avg_train_loss = loss_log['train/loss'] / num_batches
                avg_train_acc = loss_log['train/acc'] / num_batches

                print ("epoch: %d, updates: %d, time: %.2f, avg_train_loss: %.5f, avg_train_acc: %.5f" % (epoch, n_iter, \
                    time_taken, avg_train_loss, avg_train_acc))
                # writing values to SummaryWriter
                writer.add_scalar('train/loss', avg_train_loss, n_iter)
                writer.add_scalar('train/acc', avg_train_acc, n_iter)
                # reset values back
                loss_log = {'train/loss' : 0.0, 'train/acc' : 0.0}
                num_batches = 0.0
                time_start = time.time()

        val_loss, val_acc, time_taken = evaluate_model(opts, model, valid_loader, criterion)
        print ("epoch: %d, updates: %d, time: %.2f, avg_valid_loss: %.5f, avg_valid_acc: %.5f" % (epoch, n_iter, \
                time_taken, val_loss, val_acc))
        # writing values to SummaryWriter
        writer.add_scalar('val/loss', val_loss, n_iter)
        writer.add_scalar('val/acc', val_acc, n_iter)
        print ('')

        # Save the model to disk
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                'opts': opts,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
            'opts': opts,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)


if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'train':
        train(opts)
    else:
        raise NotImplementedError('unrecognized mode')
