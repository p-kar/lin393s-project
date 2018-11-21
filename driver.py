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
from utils.misc import set_random_seeds, GloveLoader
from utils.logger import TensorboardXLogger
from utils.dataset import QuoraQuestionPairsDataset, RedditCommentPairsDataset, collate_data
from utils.dataloader import MultiLoader
from models.baselines import *
from models.decomposable_attention import DecomposableAttention
from models.ESIMMultiTask import ESIMMultiTask
from models.SSEMultiTask import SSEMultiTask

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

def run_quora_iter(d, model, criterion):
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

    return loss, acc

def run_reddit_iter(d, model, criterion):
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

    return loss, prec1, prec3

def evaluate_model_quora(opts, model, loader, criterion):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for i, d in enumerate(loader):

            loss, acc = run_quora_iter(d, model, criterion)
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

            loss, prec1, prec3 = run_reddit_iter(d, model, criterion)
            val_loss += loss.data.cpu().item()
            val_prec1 += prec1.data.cpu().item()
            val_prec3 += prec3.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_prec1 = val_prec1 / num_batches
    avg_valid_prec3 = val_prec3 / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_prec1, avg_valid_prec3, time_taken

def train_quora(opts):

    glove_loader = GloveLoader(glove_emb_file=opts.glove_emb_file)

    train_dataset = QuoraQuestionPairsDataset(opts.data_dir, split='train', \
        glove_loader=glove_loader, maxlen=opts.maxlen)
    valid_dataset = QuoraQuestionPairsDataset(opts.data_dir, split='val', \
        glove_loader=glove_loader, maxlen=opts.maxlen)

    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    if opts.arch == 'lstm_concat':
        model = LSTMWithConcatBaseline(hidden_size=opts.hidden_size, num_layers=opts.num_layers, \
            bidirectional=opts.bidirectional, glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'lstm_dist_angle':
        model = LSTMWithDistAngleBaseline(hidden_size=opts.hidden_size, num_layers=opts.num_layers, \
            bidirectional=opts.bidirectional, glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'decomp_attention':
        model = DecomposableAttention(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'esim_multitask':
        model = ESIMMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'sse_multitask':
        model = SSEMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
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
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['loss', 'acc'])

    # for choosing the best model
    best_val_acc = 0.0

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        logger.step()
        for i, d in enumerate(train_loader):
            loss, acc = run_quora_iter(d, model, criterion)
            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()
            # log the losses
            logger.update(loss, acc)

        val_loss, val_acc, time_taken = evaluate_model_quora(opts, model, valid_loader, criterion)
        # log the validation losses
        logger.log_valid(time_taken, val_loss, val_acc)
        print ('')

        # Save the model to disk
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
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
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)

def train_reddit(opts):

    glove_loader = GloveLoader(glove_emb_file=opts.glove_emb_file)

    train_dataset = RedditCommentPairsDataset(opts.data_dir, split='train', \
        glove_loader=glove_loader, maxlen=opts.maxlen, K=opts.n_candidate_resp)
    valid_dataset = RedditCommentPairsDataset(opts.data_dir, split='val', \
        glove_loader=glove_loader, maxlen=opts.maxlen, K=opts.n_candidate_resp)

    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    if opts.arch == 'esim_multitask':
        model = ESIMMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'sse_multitask':
        model = SSEMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
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
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['loss', 'prec1', 'prec3'])

    # for choosing the best model
    best_val_prec1 = 0.0

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        logger.step()
        for i, d in enumerate(train_loader):
            
            loss, prec1, prec3 = run_reddit_iter(d, model, criterion)
            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            # log the losses
            logger.update(loss, prec1, prec3)

        val_loss, val_prec1, val_prec3, time_taken = evaluate_model_reddit(opts, model, valid_loader, criterion)
        # log the validation losses
        logger.log_valid(time_taken, val_loss, val_prec1, val_prec3)
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

def train_multitask(opts):

    glove_loader = GloveLoader(glove_emb_file=opts.glove_emb_file)

    qtrain_dataset = QuoraQuestionPairsDataset(os.path.join(opts.data_dir, 'quora'), split='train', \
        glove_loader=glove_loader, maxlen=opts.maxlen)
    qvalid_dataset = QuoraQuestionPairsDataset(os.path.join(opts.data_dir, 'quora'), split='val', \
        glove_loader=glove_loader, maxlen=opts.maxlen)

    qtrain_loader = DataLoader(qtrain_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    qvalid_loader = DataLoader(qvalid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    rtrain_dataset = RedditCommentPairsDataset(os.path.join(opts.data_dir, 'reddit'), split='train', \
        glove_loader=glove_loader, maxlen=opts.maxlen, K=opts.n_candidate_resp)
    rvalid_dataset = RedditCommentPairsDataset(os.path.join(opts.data_dir, 'reddit'), split='val', \
        glove_loader=glove_loader, maxlen=opts.maxlen, K=opts.n_candidate_resp)

    rtrain_loader = DataLoader(rtrain_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)
    rvalid_loader = DataLoader(rvalid_dataset, batch_size=opts.bsize, shuffle=opts.shuffle, \
        num_workers=opts.nworkers, pin_memory=True)

    train_loader = MultiLoader([qtrain_loader, rtrain_loader])

    if opts.arch == 'esim_multitask':
        model = ESIMMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
    elif opts.arch == 'sse_multitask':
        model = SSEMultiTask(hidden_size=opts.hidden_size, dropout_p=opts.dropout_p, \
            glove_loader=glove_loader, pretrained_emb=opts.pretrained_emb)
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
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['loss', 'acc', 'prec1', 'prec3'])

    # for choosing the best model
    best_val_acc = 0.0

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        scheduler.step()
        logger.step()
        for i, d in enumerate(train_loader):
            dq = d[0]
            dr = d[1]
            loss_q, acc = run_quora_iter(dq, model, criterion)
            loss_r, prec1, prec3 = run_reddit_iter(dr, model, criterion)
            loss = loss_q + loss_r
            # perform update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()
            # log the losses
            logger.update(loss, acc, prec1, prec3)

        val_loss_q, val_acc, time_taken_q = evaluate_model_quora(opts, model, qvalid_loader, criterion)
        val_loss_r, val_prec1, val_prec3, time_taken_r = evaluate_model_reddit(opts, model, rvalid_loader, criterion)
        val_loss = val_loss_q + val_loss_r
        time_taken = time_taken_q + time_taken_r
        # log the validation losses
        logger.log_valid(time_taken, val_loss, val_acc, val_prec1, val_prec3)
        print ('')

        # Save the model to disk
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
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
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
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
    elif opts.mode == 'train_multitask':
        train_multitask(opts)
    else:
        raise NotImplementedError('unrecognized mode')
