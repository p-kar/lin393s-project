import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("..")
from utils.misc import loadGloveFile, ixvr

class AttendFeedForward(nn.Module):
	"""
	Similiar to the attend (Section 3.1) module of the DecAtt paper
	"""
	def __init__(self, inp_size, hidden_size=200, dropout_p=0.2):
		super(AttendFeedForward, self).__init__()

		self.hidden_size = hidden_size
		self.linear = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(inp_size, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, hidden_size), \
			nn.ReLU())

	def forward(self, s1, s2, mask1, mask2):
		"""
		Args:
			s1: Sentence 1 BiLSTM embeddings (b x LA x inp_size)
			s2: Sentence 2 BiLSTM embeddings (b x LB x inp_size)
			mask1: Sentence 1 mask (b x LA)
			mask2: Sentence 2 mask (b x LB)
		Output:
			alphas: Soft aligned combinations of s1 w.r.t. s2 tokens (b x maxlen x inp_size)
			betas: Soft aligned combinations of s2 w.r.t. s1 tokens (b x maxlen x inp_size)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]
		inp_size = s1.shape[2]

		h1 = self.linear(s1.view(-1, inp_size)).view(batch_size, maxlen, -1)
		# b x LA x hidden_size
		h2 = self.linear(s2.view(-1, inp_size)).view(batch_size, maxlen, -1)
		# b x LB x hidden_size
		h2t = torch.transpose(h2, 1, 2)
		# b x hidden_size x LB

		e = torch.bmm(h1, h2t)

		e_alpha = torch.mul(e, mask1.unsqueeze(-1))
		e_alpha = torch.exp(e_alpha - torch.max(e_alpha, dim=1)[0].unsqueeze(1))
		e_alpha = torch.div(e_alpha, torch.sum(e_alpha, dim=1).unsqueeze(1))
		# b x LA x LB

		e_beta = torch.mul(e, mask2.unsqueeze(1))
		e_beta = torch.exp(e_beta - torch.max(e_beta, dim=2)[0].unsqueeze(-1))
		e_beta = torch.div(e_beta, torch.sum(e_beta, dim=2).unsqueeze(-1))
		# b x LA x LB

		alphas = torch.bmm(torch.transpose(e_alpha, 1, 2), s1)
		alphas = torch.mul(alphas, mask2.unsqueeze(-1))
		# b x LB x inp_size
		betas = torch.bmm(e_beta, s2)
		betas = torch.mul(betas, mask1.unsqueeze(-1))
		# b x LA x inp_size

		return alphas, betas

class CompareFeedForward(nn.Module):
	"""
	Similar to the compare (Section 3.2) module of the DecAtt paper
	except instead of returning the sum of the embeddings v1 and v2
	(which might be susceptible to the length of the sequence),
	this returns v1_avg, v1_max, v2_avg, v2_max.
	"""
	def __init__(self, inp_size, hidden_size=200, dropout_p=0.2):
		super(CompareFeedForward, self).__init__()

		self.linear = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(inp_size * 2, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, hidden_size), \
			nn.ReLU())

	def forward(self, s1, s2, alphas, betas, mask1, mask2):
		"""
		Args:
			s1: Sentence 1 BiLSTM embeddings (b x LA x inp_size)
			s2: Sentence 2 BiLSTM embeddings (b x LB x inp_size)
			alphas: Aligned phrases (b x LB x inp_size)
			betas: Aligned phrases (b x LA x inp_size)
			mask1: Sentence 1 mask (b x LA)
			mask2: Sentence 2 mask (b x LB)
		Output:
			v1_avg: Comparison avg. pooled vector for aligned sentence s1 (b x hidden_size)
			v1_max: Comparison max. pooled vector for aligned sentence s1 (b x hidden_size)
			v2_avg: Comparison avg. pooled vector for aligned sentence s2 (b x hidden_size)
			v2_max: Comparison max. pooled vector for aligned sentence s2 (b x hidden_size)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]
		inp_size = s1.shape[2]

		in1 = torch.cat((s1, betas), dim=2)
		# b x LA x (inp_size * 2)
		in2 = torch.cat((s2, alphas), dim=2)
		# b x LB x (inp_size * 2)

		v1 = self.linear(in1.view(-1, inp_size * 2)).view(batch_size, maxlen, -1)
		# b x LA x hidden_size
		v1_avg = torch.sum(torch.mul(v1, mask1.unsqueeze(-1)), dim=1)
		v1_avg = torch.div(v1_avg, torch.sum(mask1, dim=1).unsqueeze(-1))
		# b x hidden_size
		v1_max = torch.max(torch.mul(v1, mask1.unsqueeze(-1)), dim=1)[0]
		# b x hidden_size

		v2 = self.linear(in2.view(-1, inp_size * 2)).view(batch_size, maxlen, -1)
		# b x LB x hidden_size
		v2_avg = torch.sum(torch.mul(v2, mask2.unsqueeze(-1)), dim=1)
		v2_avg = torch.div(v2_avg, torch.sum(mask2, dim=1).unsqueeze(-1))
		# b x hidden_size
		v2_max = torch.max(torch.mul(v2, mask2.unsqueeze(-1)), dim=1)[0]
		# b x hidden_size

		return v1_avg, v1_max, v2_avg, v2_max

class ESIMMultiTask(nn.Module):
	"""
	Model architecture similar to the Enhanced Sequential Inference Model (ESIM)
	as described in https://arxiv.org/abs/1609.06038 without the Tree LSTM.

	The model is designed for both Reddit response prediction task and Quora
	semantic question matching task.
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
		super(ESIMMultiTask, self).__init__()

		if not pretrained_emb:
			raise NotImplementedError('always loads pretrained embeddings')

		_, _, word_vectors = loadGloveFile(glove_emb_file)
		word_vectors = np.vstack(word_vectors)
		vocab_size = word_vectors.shape[0]
		embed_size = word_vectors.shape[1]

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
		self.drop = nn.Dropout(p=dropout_p)
		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
		self.attend = AttendFeedForward(inp_size=hidden_size * 2, hidden_size=hidden_size, dropout_p=dropout_p)
		self.compare = CompareFeedForward(inp_size=hidden_size * 2, hidden_size=hidden_size, dropout_p=dropout_p)

		# prediction layer for the Quora task
		self.sts_pred = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 4, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, 2))

		# tranformation layer for the response
		self.response_transform = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 2, hidden_size * 2), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 2, hidden_size * 2))

		self.reset_parameters()

	def reset_parameters(self):
		"""Initialize network weights using Xavier init (with bias 0.01)"""

		self. apply(ixvr)

	def forward(self, s1, s2, len1, len2):
		"""
		Args:
			s1: Sentence 1 embeddings (b x LA)
			s2: Sentence 2 embeddings (b x LB)
			len1: Sentence 1 length (b)
			len2: Sentence 2 length (b)
		"""
		batch_size = s1.shape[0]
		maxlen = s1.shape[1]

		s1 = self.drop(self.embedding(s1)).transpose(0, 1)
		s1, _ = self.encoder(s1)
		s1 = torch.transpose(s1, 0, 1).contiguous()
		# b x LA x (hidden_size * 2)
		s2 = self.drop(self.embedding(s2)).transpose(0, 1)
		s2, _ = self.encoder(s2)
		s2 = torch.transpose(s2, 0, 1).contiguous()
		# b x LB x (hidden_size * 2)

		mask1 = torch.arange(0, maxlen).expand(batch_size, maxlen)
		if torch.cuda.is_available():
			mask1 = mask1.cuda()
		mask1 = mask1 < len1.unsqueeze(-1)
		mask2 = torch.arange(0, maxlen).expand(batch_size, maxlen)
		if torch.cuda.is_available():
			mask2 = mask2.cuda()
		mask2 = mask2 < len2.unsqueeze(-1)

		mask1 = mask1.float()
		mask2 = mask2.float()

		alphas, betas = self.attend(s1, s2, mask1, mask2)
		v1_avg, v1_max, v2_avg, v2_max = self.compare(s1, s2, alphas, betas, mask1, mask2)
		out = self.sts_pred(torch.cat((v1_avg, v1_max, v2_avg, v2_max), dim=1))

		return out

	def rank_responses(self, q, resp, len_q, len_resp):
		"""
		Args:
			q: Reddit question embeddings (b x LA)
			resp: Reddit response candidates embeddings (b x K x LB)
			len_q: Length of the input question (b)
			len_resp: Length of the response candidates (b x K)
		"""

		batch_size = q.shape[0]
		maxlen = q.shape[1]
		K = resp.shape[1]

		q = self.drop(self.embedding(q)).transpose(0, 1)
		q, _ = self.encoder(q)
		q = torch.transpose(q, 0, 1).contiguous()
		# b x LA x (hidden_size * 2)
		resp = self.drop(self.embedding(resp)).view(batch_size * K, maxlen, -1).transpose(0, 1)
		resp, _ = self.encoder(resp)
		resp = torch.transpose(resp, 0, 1).view(batch_size, K, maxlen, -1).contiguous()
		# b x K x LB x (hidden_size * 2)

		mask1 = torch.arange(0, maxlen).expand(batch_size, maxlen)
		if torch.cuda.is_available():
			mask1 = mask1.cuda()
		mask1 = mask1 < len_q.unsqueeze(-1)
		mask1 = mask1.float()
		# b x LA

		mask2 = torch.arange(0, maxlen).expand(batch_size * K, maxlen)
		if torch.cuda.is_available():
			mask2 = mask2.cuda()
		mask2 = mask2 < len_resp.view(-1).unsqueeze(-1)
		mask2 = mask2.view(batch_size, K, -1).float()
		# b x K x LB

		q = q.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(batch_size * K, maxlen, -1)
		# (b * K) x LA x (hidden_size * 2)
		mask1 = mask1.unsqueeze(1).expand(-1, K, -1).contiguous().view(batch_size * K, maxlen)
		# (b * K) x LA

		resp = resp.view(batch_size * K, maxlen, -1)
		# (b * K) x LB x (hidden_size * 2)
		mask2 = mask2.view(batch_size * K, maxlen)
		# (b * K) x LB

		alphas, betas = self.attend(q, resp, mask1, mask2)
		v1_avg, v1_max, v2_avg, v2_max = self.compare(q, resp, alphas, betas, mask1, mask2)

		v1 = torch.cat((v1_avg, v1_max), dim=1)
		# (b * K) x (hidden_size * 2)
		v2 = self.response_transform(torch.cat((v2_avg, v2_max), dim=1))
		# (b * K) x (hidden_size * 2)

		scores = torch.sum(torch.mul(v1, v2), dim=1).view(batch_size, -1)
		# b x K

		return scores
