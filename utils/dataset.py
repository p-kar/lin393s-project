import os
import csv
import pdb
import numpy as np
import random
from nltk import word_tokenize

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .misc import loadGloveFile

def readQuoraDataFile(fname):
	"""
	Args:
		fname: file containing the sentence pairs for the split
	Output:
		samples: Triplets containing 2 sentences and the label
	"""

	with open(fname, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		content = [row for row in reader]

	samples = []
	for line in content:
		if len(line) == 6:
			line = line[-3:]
			line[0] = word_tokenize(line[0], preserve_line=True)
			line[1] = word_tokenize(line[1], preserve_line=True)
			line[2] = int(line[2])
			if len(line[0]) == 0 or len(line[1]) == 0:
				continue
			samples.append(line)

	return samples

def readRedditDataFile(fname):
	"""
	Args:
		fname: file containing the sentence pairs for the split
	Output:
		samples: Pairs containing 2 sentences (comment and its response)
	"""

	with open(fname, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		content = [row for row in reader]

	samples = []
	for line in content:
		if len(line) == 2:
			line[0] = word_tokenize(line[0], preserve_line=True)
			line[1] = word_tokenize(line[1], preserve_line=True)
			if len(line[0]) == 0 or len(line[1]) == 0:
				continue
			samples.append(line)

	return samples

def collate_data(batch):
	if isinstance(batch[0], str):
		return batch
	else:
		return default_collate(batch)

class QuoraQuestionPairsDataset(Dataset):
	"""Quora Question Pairs Dataset"""

	def __init__(self, root='./data/quora', split='train', glove_emb_file='./data/glove.6B/glove.6B.50d.txt', \
		maxlen=30):
		
		self.word_to_index, self.index_to_word, self.word_vectors = loadGloveFile(glove_emb_file)
		self.split = split
		self.glove_vec_size = self.word_vectors[0].shape[0]
		self.data_file = os.path.join(root, split + '.csv')
		self.data = readQuoraDataFile(self.data_file)
		self.maxlen = maxlen

	def __len__(self):
		return len(self.data)

	def _parse(self, sent):
		sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
		sent = sent[:self.maxlen]
		padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
		sent.extend(padding)
		return np.array([self.word_to_index[s] for s in sent])

	def __getitem__(self, idx):
		raw_s1 = ' '.join(self.data[idx][0])
		raw_s2 = ' '.join(self.data[idx][1])
		label = self.data[idx][2]
		s1 = torch.LongTensor(self._parse(self.data[idx][0]))
		s2 = torch.LongTensor(self._parse(self.data[idx][1]))
		len_s1 = min(self.maxlen, len(self.data[idx][0]))
		len_s2 = min(self.maxlen, len(self.data[idx][1]))

		return {'s1': s1, 's2': s2, 'raw_s1': raw_s1, \
		'raw_s2': raw_s2, 'label': label, 'len1': len_s1, \
		'len2': len_s2}

class RedditCommentPairsDataset(Dataset):
	"""Reddit Comment Pairs Dataset"""

	def __init__(self, root='./data/reddit', split='train', glove_emb_file='./data/glove.6B/glove.6B.50d.txt', \
		maxlen=30, K=10):

		self.word_to_index, self.index_to_word, self.word_vectors = loadGloveFile(glove_emb_file)
		self.split = split
		self.glove_vec_size = self.word_vectors[0].shape[0]
		self.data_file = os.path.join(root, split + '.csv')
		self.data = readRedditDataFile(self.data_file)
		self.maxlen = maxlen
		self.K = K
		self.sample_indexes = np.arange(0, self.__len__())
		np.random.shuffle(self.sample_indexes)
		assert self.K < self.__len__()

	def __len__(self):
		return len(self.data)

	def _parse(self, sent):
		sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
		sent = sent[:self.maxlen]
		padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
		sent.extend(padding)
		return np.array([self.word_to_index[s] for s in sent])

	def _get_random_comments(self, correct):
		start_idx = random.randint(0, self.__len__() - self.K)
		indexes = self.sample_indexes[start_idx:start_idx + self.K - 1]

		sents = [self.data[idx][1] for idx in indexes]
		label = random.randint(0, self.K - 1)
		sents = sents[:label] + [correct] + sents[label:]
		resp = torch.stack([torch.LongTensor(self._parse(s)) for s in sents])
		len_resp = torch.LongTensor([min(self.maxlen, len(s)) for s in sents])

		return resp, len_resp, label

	def __getitem__(self, idx):
		q = torch.LongTensor(self._parse(self.data[idx][0]))
		len_q = min(self.maxlen, len(self.data[idx][0]))
		resp, len_resp, label = self._get_random_comments(self.data[idx][1])

		return {'q': q, 'resp': resp, 'label': label, 'len_q': len_q, 'len_resp': len_resp}
