import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("..")
from utils.misc import ixvr

class SSEMultiTask(nn.Module):
	"""
	Model architecture similar to the Shortcut-Stacked Sentence Encoder as
	described in https://arxiv.org/pdf/1708.02312.pdf.

	The model is designed for both Reddit response prediction task and Quora
	semantic question matching task.
	"""
	def __init__(self, hidden_size=200, dropout_p=0.2, \
		glove_loader=None, pretrained_emb=True):
		"""
		Args:
			hidden_size: Size of the intermediate linear layers
			dropout_p: Dropout probability for intermediate dropout layers
			glove_loader: GLoVe embedding loader
			pretrained_emb: Use pretrained embeddings
		"""
		super(SSEMultiTask, self).__init__()

		if not pretrained_emb:
			raise NotImplementedError('always loads pretrained embeddings')

		word_vectors = glove_loader.word_vectors
		word_vectors = np.vstack(word_vectors)
		vocab_size = word_vectors.shape[0]
		embed_size = word_vectors.shape[1]

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
		
		self.encoder1 = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))
		self.encoder2 = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.LSTM(input_size=embed_size + 2*hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))
		self.encoder3 = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.LSTM(input_size=embed_size + 4*hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))

		# prediction layer for the Quora task
		self.sts_pred = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 8, hidden_size), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size, 2))

		# tranformation layer for the response
		self.response_transform = nn.Sequential( \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 4, hidden_size * 4), \
			nn.ReLU(), \
			nn.Dropout(p=dropout_p), \
			nn.Linear(hidden_size * 4, hidden_size * 4))

		self.reset_parameters()

	def reset_parameters(self):
		"""Initialize network weights using Xavier init (with bias 0.01)"""

		self. apply(ixvr)

	def _encode(self, s, len_s):
		"""
		Args:
			s: Tokenized sentence (b x L)
			len_s: Sentence length (b)
		Output:
			out: Output vector with concatenated avg. and max. pooled
				sentence encoding (b x (hidden_size * 4))
		"""
		batch_size = s.shape[0]
		maxlen = s.shape[1]

		s = self.embedding(s).transpose(0, 1)
		# L x b x embed_size
		h1, _ = self.encoder1(s)
		h2, _ = self.encoder2(torch.cat((s, h1), dim=2))
		h3, _ = self.encoder3(torch.cat((s, h1, h2), dim=2))
		v = torch.transpose(h3, 0, 1)
		# b x L x (hidden_size * 2)

		mask = torch.arange(0, maxlen).expand(batch_size, maxlen)
		if torch.cuda.is_available():
			mask = mask.cuda()
		mask = mask < len_s.unsqueeze(-1)
		mask = mask.float()

		v_avg = torch.sum(torch.mul(v, mask.unsqueeze(-1)), dim=1)
		v_avg = torch.div(v_avg, torch.sum(mask, dim=1).unsqueeze(-1))
		# b x (hidden_size * 2)
		v_max = torch.max(torch.mul(v, mask.unsqueeze(-1)), dim=1)[0]
		# b x (hidden_size * 2)

		out = torch.cat((v_avg, v_max), dim=1)
		# b x (hidden_size * 4)

		return out

	def forward(self, s1, s2, len1, len2):
		"""
		Args:
			s1: Tokenized sentence 1 (b x LA)
			s2: Tokenized sentence 2 (b x LB)
			len1: Sentence 1 length (b)
			len2: Sentence 2 length (b)
		"""
		v1 = self._encode(s1, len1)
		# b x LA x (hidden_size * 4)
		v2 = self._encode(s2, len2)
		# b x LB x (hidden_size * 4)
		out = self.sts_pred(torch.cat((v1, v2), dim=1))
		# b x 2

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

		vq = self._encode(q, len_q)
		# b x (hidden_size * 4)
		vresp = self._encode(resp.view(-1, maxlen), len_resp.view(-1))
		# (b * K) x (hidden_size * 4)
		vresp = self.response_transform(vresp).view(batch_size, K, -1)
		# b x K x (hidden_size * 4)

		scores = torch.sum(torch.mul(vq.unsqueeze(1), vresp), dim=2)
		# b x K

		return scores
