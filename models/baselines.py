import pdb
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append("..")
from utils.misc import loadGloveFile

class LSTMWithConcatBaseline(nn.Module):

	def __init__(self, hidden_size=300, num_layers=1, bidirectional=False, \
		glove_loader=None, pretrained_emb=True):
		"""
		"""
		super(LSTMWithConcatBaseline, self).__init__()
		if not pretrained_emb:
			raise NotImplementedError('always loads pretrained embeddings')
		if num_layers > 1:
			raise NotImplementedError('don\'t support multiple LSTM cell')
		if bidirectional:
			raise NotImplementedError('don\'t support bidirectional LSTMs')

		word_vectors = glove_loader.word_vectors
		word_vectors = np.vstack(word_vectors)
		vocab_size = word_vectors.shape[0]
		embed_size = word_vectors.shape[1]

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
		self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, \
			bidirectional=bidirectional)
		self.linear = nn.Linear(hidden_size * 2, 2)

	def forward(self, s1, s2, len1, len2):
		"""
		Args:
			s1: Sentence 1 tokenized with embedding index (b x maxlen)
			s2: Sentence 2 tokenized with embedding index (b x maxlen)
			len1: Sentence 1 length (b)
			len2: Sentence 2 length (b)
		Output:
			out: (b x 2)
		"""

		batch_size = s1.shape[0]

		embed_s1 = torch.transpose(self.embedding(s1), 0, 1)
		# maxlen x b x embed_size
		embed_s2 = torch.transpose(self.embedding(s2), 0, 1)
		# maxlen x b x embed_size

		out_s1, _ = self.rnn(embed_s1)
		out_s1 = torch.transpose(out_s1, 0, 1)
		# b x maxlen x hidden_size
		out_s1 = out_s1[torch.arange(0, batch_size), len1 - 1, :]
		# b x hidden_size

		out_s2, _ = self.rnn(embed_s2)
		out_s2 = torch.transpose(out_s2, 0, 1)
		# b x maxlen x hidden_size
		out_s2 = out_s2[torch.arange(0, batch_size), len2 - 1, :]

		concat = torch.cat((out_s1, out_s2), dim=1)
		# b x (hidden_size * 2)
		out = self.linear(concat)

		return out

class LSTMWithDistAngleBaseline(nn.Module):

	def __init__(self, hidden_size=300, num_layers=1, bidirectional=False, \
		glove_loader=None, pretrained_emb=True):
		"""
		"""
		super(LSTMWithDistAngleBaseline, self).__init__()
		if num_layers > 1:
			raise NotImplementedError('don\'t support multiple LSTM cell')
		if bidirectional:
			raise NotImplementedError('don\'t support bidirectional LSTMs')

		word_vectors = glove_loader.word_vectors
		word_vectors = np.vstack(word_vectors)
		vocab_size = word_vectors.shape[0]
		embed_size = word_vectors.shape[1]

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
		self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, \
			bidirectional=bidirectional)
		self.linear = nn.Linear(hidden_size + 1, 2)
		self.cosine = nn.modules.distance.CosineSimilarity()

	def forward(self, s1, s2, len1, len2):
		"""
		Args:
			s1: Sentence 1 tokenized with embedding index (b x maxlen)
			s2: Sentence 2 tokenized with embedding index (b x maxlen)
			len1: Sentence 1 length (b)
			len2: Sentence 2 length (b)
		Output:
			out: (b x 2)
		"""

		batch_size = s1.shape[0]

		embed_s1 = torch.transpose(self.embedding(s1), 0, 1)
		# maxlen x b x embed_size
		embed_s2 = torch.transpose(self.embedding(s2), 0, 1)
		# maxlen x b x embed_size

		out_s1, _ = self.rnn(embed_s1)
		out_s1 = torch.transpose(out_s1, 0, 1)
		# b x maxlen x hidden_size
		out_s1 = out_s1[torch.arange(0, batch_size), len1 - 1, :]
		# b x hidden_size

		out_s2, _ = self.rnn(embed_s2)
		out_s2 = torch.transpose(out_s2, 0, 1)
		# b x maxlen x hidden_size
		out_s2 = out_s2[torch.arange(0, batch_size), len2 - 1, :]

		dist_s1_s2 = torch.abs(out_s1 - out_s2)
		cos_s1_s2 = self.cosine(out_s1, out_s2)

		concat = torch.cat((dist_s1_s2, cos_s1_s2.view(-1, 1)), dim=1)
		# b x (hidden_size + 1)
		out = self.linear(concat)

		return out

