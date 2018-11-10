import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("..")
from utils.misc import loadGloveFile

class AttendFeedForward(nn.Module):
	def __init__(self, embed_size=300, hidden_size=200, dropout_p=0.2):
		"""
		As discussed in Section 3.1 of the paper
		"""
		super(AttendFeedForward, self).__init__()

		self.hidden_size = hidden_size
		self.linear = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(embed_size, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, hidden_size), \
			nn.ReLU())

	def forward(self, s1, s2, mask1, mask2):
		"""
		Args:
			s1: Sentence 1 embeddings (b x LA x embed_size)
			s2: Sentence 2 embeddings (b x LB x embed_size)
			mask1: Sentence 1 mask (b x LA)
			mask2: Sentence 2 mask (b x LB)
		Output:
			alphas: Soft aligned combinations of s1 w.r.t. s2 tokens (b x maxlen x embed_size)
			betas: Soft aligned combinations of s2 w.r.t. s1 tokens (b x maxlen x embed_size)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]
		embed_size = s1.shape[2]

		h1 = self.linear(s1.view(-1, embed_size)).view(batch_size, maxlen, -1)
		# b x LA x hidden_size
		h2 = self.linear(s2.view(-1, embed_size)).view(batch_size, maxlen, -1)
		# b x LB x hidden_size
		h2t = torch.transpose(h2, 1, 2)
		# b x hidden_size x LB

		e = torch.bmm(h1, h2t)

		e_alpha = torch.exp(e - torch.max(e, dim=2)[0].unsqueeze(1))
		e_alpha = torch.mul(e_alpha, mask1.unsqueeze(-1))
		e_alpha = torch.div(e_alpha, torch.sum(e_alpha, dim=1).unsqueeze(1))
		# b x LA x LB

		e_beta = torch.exp(e - torch.max(e, dim=1)[0].unsqueeze(-1))
		e_beta = torch.mul(e_beta, mask2.unsqueeze(1))
		e_beta = torch.div(e_beta, torch.sum(e_beta, dim=2).unsqueeze(-1))
		# b x LA x LB

		alphas = torch.bmm(torch.transpose(e_alpha, 1, 2), s1)
		alphas = torch.mul(alphas, mask2.unsqueeze(-1))
		# b x LB x embed_size
		betas = torch.bmm(e_beta, s2)
		betas = torch.mul(betas, mask1.unsqueeze(-1))
		# b x LA x embed_size

		return alphas, betas

class CompareFeedForward(nn.Module):
	def __init__(self, embed_size=300, hidden_size=200, dropout_p=0.2):
		"""
		As discussed in Section 3.2 of the paper
		"""
		super(CompareFeedForward, self).__init__()

		self.linear = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(embed_size * 2, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, hidden_size), \
			nn.ReLU())

	def forward(self, s1, s2, alphas, betas, mask1, mask2):
		"""
		Args:
			s1: Sentence 1 embeddings (b x LA x embed_size)
			s2: Sentence 2 embeddings (b x LB x embed_size)
			alphas: Aligned phrases (b x LB x embed_size)
			betas: Aligned phrases (b x LA x embed_size)
			mask1: Sentence 1 mask (b x LA)
			mask2: Sentence 2 mask (b x LB)
		Output:
			v1: Comparison vector for aligned sentence s1 (b x hidden_size)
			v2: Comparison vector for aligned sentence s2 (b x hidden_size)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]
		embed_size = s1.shape[2]

		in1 = torch.cat((s1, betas), dim=2)
		# b x LA x (embed_size * 2)
		in2 = torch.cat((s2, alphas), dim=2)
		# b x LB x (embed_size * 2)

		v1 = self.linear(in1.view(-1, embed_size * 2)).view(batch_size, maxlen, -1)
		# b x LA x hidden_size
		v1 = torch.sum(torch.mul(v1, mask1.unsqueeze(-1)), dim=1)
		# b x hidden_size
		v2 = self.linear(in2.view(-1, embed_size * 2)).view(batch_size, maxlen, -1)
		# b x LB x hidden_size
		v2 = torch.sum(torch.mul(v2, mask2.unsqueeze(-1)), dim=1)
		# b x hidden_size

		return v1, v2

class AggregateFeedForward(nn.Module):
	def __init__(self, hidden_size=200, dropout_p=0.2):
		"""
		As discussed in Section 3.3 of the paper
		"""
		super(AggregateFeedForward, self).__init__()

		self.linear = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 2, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, 2))

	def forward(self, v1, v2):
		"""
		Args:
			v1: Comparison vector for aligned sentence s1 (b x hidden_size)
			v2: Comparison vector for aligned sentence s2 (b x hidden_size)
		Output:

		"""
		inp = torch.cat((v1, v2), dim=1)
		y_hat = self.linear(inp)

		return y_hat

class DecomposableAttention(nn.Module):
	"""
	Implementation of Decomposable Attention Model as described in
	https://arxiv.org/pdf/1606.01933.pdf
	"""
	def __init__(self, hidden_size=200, dropout_p=0.2, \
		glove_emb_file='./data/glove.6B/glove.6B.50d.txt', pretrained_emb=True):
		"""
		Args:
			hidden_size: Size of the intermediate linear layers
			dropout_p: Dropout probability for intermediate dropout layers
			glove_emb_file: Location of the pretrained GloVe embeddings
			pretrained_emb: Use pretrained embeddings
		"""
		super(DecomposableAttention, self).__init__()

		if not pretrained_emb:
			raise NotImplementedError('always loads pretrained embeddings')

		_, _, word_vectors = loadGloveFile(glove_emb_file)
		word_vectors = np.vstack(word_vectors)
		vocab_size = word_vectors.shape[0]
		embed_size = word_vectors.shape[1]

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
		self.attend = AttendFeedForward(embed_size=embed_size, hidden_size=hidden_size, dropout_p=dropout_p)
		self.compare = CompareFeedForward(embed_size=embed_size, hidden_size=hidden_size, dropout_p=dropout_p)
		self.aggregate = AggregateFeedForward(hidden_size=hidden_size, dropout_p=dropout_p)

	def forward(self, s1, s2, len1, len2):
		"""
		Args:
			s1: Sentence 1 embeddings (b x LA)
			s2: Sentence 2 embeddings (b x LB)
			len1: Sentence 1 mask (b)
			len2: Sentence 2 mask (b)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]

		s1 = self.embedding(s1)
		# b x LA x embed_size
		s2 = self.embedding(s2)
		# b x LB x embed_size

		mask1 = torch.arange(0, maxlen).expand(batch_size, maxlen)
		mask1 = mask1 < len1.unsqueeze(-1)
		mask2 = torch.arange(0, maxlen).expand(batch_size, maxlen)
		mask2 = mask2 < len2.unsqueeze(-1)

		if torch.cuda.is_available():
			mask1 = mask1.float().cuda()
			mask2 = mask2.float().cuda()
		else:
			mask1 = mask1.float()
			mask2 = mask2.float()

		alphas, betas = self.attend(s1, s2, mask1, mask2)
		v1, v2 = self.compare(s1, s2, alphas, betas, mask1, mask2)
		out = self.aggregate(v1, v2)

		return out

