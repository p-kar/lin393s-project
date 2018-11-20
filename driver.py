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
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.arguments import get_args
from utils.misc import set_random_seeds
from utils.dataset import QuoraQuestionPairsDataset, RedditCommentPairsDataset, collate_data
from models.baselines import *
from models.decomposable_attention import DecomposableAttention
from models.ESIMMultiTask import ESIMMultiTask

use_cuda = torch.cuda.is_available()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1. / batch_size))
    return res

def evaluate_model_quora(opts, model, loader, criterion):
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

def evaluate_model_reddit(opts, model, loader, criterion):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_prec1 = 0.0
    val_prec3 = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for i, d in enumerate(loader):
            batch_size = d['q'].shape[0]
            if use_cuda:
                d['q'] = d['q'].cuda()
                d['resp'] = d['resp'].cuda()
                d['len_q'] = d['len_q'].cuda()
                d['len_resp'] = d['len_resp'].cuda()
                d['label'] = d['label'].cuda()

            out = model.rank_responses(d['q'], d['resp'], d['len_q'], d['len_resp'])
            loss = criterion(out, d['label'])
            prec1, prec3 = accuracy(out, d['label'], topk=(1, 3))

            val_loss += loss.data.cpu().item()
            val_prec1 += prec1.data.cpu().item()
            val_prec3 += prec3.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_prec1 = val_prec1 / num_batches
    avg_valid_prec3 = val_prec3 / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_prec1, avg_train_prec3, time_taken

def train_quora(opts):

    train_dataset = QuoraQuestionPairsDataset(opts.data_dir, split='train', \
        glove_emb_file=opts.glove_emb_file, maxlen=opts.maxlen)
    valid_dataset = QuoraQuestionPairsDataset(opts.data_dir, split='val', \
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
    elif opts.arch == 'decomp_attention':
        model = DecomposableAttention(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_emb_file=opts.glove_emb_file, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'esim_multitask':
        model = ESIMMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_emb_file=opts.glove_emb_file, pretrained_emb=opts.pretrained_emb)
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

        val_loss, val_acc, time_taken = evaluate_model_quora(opts, model, valid_loader, criterion)
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

def train_reddit(opts):

    train_dataset = RedditCommentPairsDataset(opts.data_dir, split='train', \
        glove_emb_file=opts.glove_emb_file, maxlen=opts.maxlen, K=opts.n_candidate_resp)
    valid_dataset = RedditCommentPairsDataset(opts.data_dir, split='val', \
        glove_emb_file=opts.glove_emb_file, maxlen=opts.maxlen, K=opts.n_candidate_resp)

    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    if opts.arch == 'esim_multitask':
        model = ESIMMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_emb_file=opts.glove_emb_file, pretrained_emb=opts.pretrained_emb)
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
    loss_log = {'train/loss' : 0.0, 'train/acc (top 1)' : 0.0, 'train/acc (top 3)' : 0.0}
    time_start = time.time()
    num_batches = 0

    # for choosing the best model
    best_val_prec1 = 0.0

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        for i, d in enumerate(train_loader):
            batch_size = d['q'].shape[0]
            if use_cuda:
                d['q'] = d['q'].cuda()
                d['resp'] = d['resp'].cuda()
                d['len_q'] = d['len_q'].cuda()
                d['len_resp'] = d['len_resp'].cuda()
                d['label'] = d['label'].cuda()

            out = model.rank_responses(d['q'], d['resp'], d['len_q'], d['len_resp'])
            loss = criterion(out, d['label'])
            prec1, prec3 = accuracy(out, d['label'], topk=(1, 3))

            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            # log the losses
            n_iter += 1
            num_batches += 1
            loss_log['train/loss'] += loss.data.cpu().item()
            loss_log['train/acc (top 1)'] += prec1.data.cpu().item()
            loss_log['train/acc (top 3)'] += prec3.data.cpu().item()

            if num_batches != 0 and n_iter % opts.log_iter == 0:
                time_end = time.time()
                time_taken = time_end - time_start
                avg_train_loss = loss_log['train/loss'] / num_batches
                avg_train_prec1 = loss_log['train/acc (top 1)'] / num_batches
                avg_train_prec3 = loss_log['train/acc (top 3)'] / num_batches

                print ("epoch: %d, updates: %d, time: %.2f, train_loss: %.5f, train_prec1: %.5f, train_prec3: %.5f" % (epoch, n_iter, \
                    time_taken, avg_train_loss, avg_train_prec1, avg_train_prec3))
                # writing values to SummaryWriter
                writer.add_scalar('train/loss', avg_train_loss, n_iter)
                writer.add_scalar('train/acc (top 1)', avg_train_prec1, n_iter)
                writer.add_scalar('train/acc (top 3)', avg_train_prec3, n_iter)
                # reset values back
                loss_log = {'train/loss' : 0.0, 'train/acc (top 1)' : 0.0, 'train/acc (top 3)' : 0.0}
                num_batches = 0.0
                time_start = time.time()

        val_loss, val_prec1, val_prec3, time_taken = evaluate_model_reddit(opts, model, valid_loader, criterion)
        print ("epoch: %d, updates: %d, time: %.2f, valid_loss: %.5f, valid_prec1: %.5f, valid_prec3: %.5f" % (epoch, n_iter, \
                time_taken, val_loss, val_prec1, val_prec3))
        # writing values to SummaryWriter
        writer.add_scalar('val/loss', val_loss, n_iter)
        writer.add_scalar('val/acc (top 1)', val_prec1, n_iter)
        writer.add_scalar('val/acc (top 3)', val_prec3, n_iter)
        print ('')

        # Save the model to disk
        if val_prec1 >= best_val_prec1:
            best_val_prec1 = val_prec1
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                'opts': opts,
                'val_prec1': val_prec1,
                'best_val_prec1': best_val_prec1
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
            'opts': opts,
            'val_prec1': val_prec1,
            'best_val_prec1': best_val_prec1
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'train_quora':
        opts.data_dir = os.path.join(opts.data_dir, 'quora')
        train_quora(opts)
    elif opts.mode == 'train_reddit':
        opts.data_dir = os.path.join(opts.data_dir, 'reddit')
        train_reddit(opts)
    else:
        raise NotImplementedError('unrecognized mode')
