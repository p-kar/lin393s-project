import pdb
import torch
import random
import numpy as np

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def loadGloveFile(fname):
	"""
	Args:
		fname: File containing the GloVe vectors
	Output:
		word_to_index: dictionary mapping from word to emb. index
		index_to_word: dictionary mapping from emb. index to word
		word_vectors: list of GloVe vectors
	"""

	with open(fname, 'r') as f:
		content = f.readlines()

	word_to_index = {}
	index_to_word = {}
	word_vectors = []

	for idx, line in enumerate(content):
		line = line.strip().split()
		word, vec = line[0], line[1:]
		vec = np.array([float(v) for v in vec])
		word_to_index[word] = idx
		index_to_word[idx] = word
		word_vectors.append(vec)

	extra_words = ['<sos>', '<eos>', '<pad>', '<unk>']
	num_words = len(word_vectors)
	glove_vec_size = word_vectors[0].shape[0]

	for word in extra_words:
		word_to_index[word] = num_words
		index_to_word[num_words] = word
		word_vectors.append(np.random.randn(glove_vec_size))
		num_words += 1

	return word_to_index, index_to_word, word_vectors

