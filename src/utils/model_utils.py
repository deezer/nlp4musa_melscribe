import torch
from tqdm import trange

from sentence_transformers import SentenceTransformer, util, models

def get_sentence_transformer_embeddings(model_name, corpus):
	corpus_chunk_size = 1000
	model = SentenceTransformer(model_name)

	text_embs = []
	if len(corpus) < corpus_chunk_size:
		text_embs = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True).detach().cpu()
	else:
		for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc="Corpus Chunks", disable=False):
        	
			corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
			sub_corpus_embeddings =  model.encode(corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, convert_to_tensor=True)
			text_embs.extend(sub_corpus_embeddings.detach().cpu())
		text_embs = torch.stack(text_embs)
	return text_embs


def get_transformer_embeddings(model_name, corpus):
	corpus_chunk_size = 1000
	word_embedding_model = models.Transformer('bert-large-cased', max_seq_length=256)
	pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
	model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

	text_embs = []
	if len(corpus) < corpus_chunk_size:
		text_embs = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True).detach().cpu()
	else:
		for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc="Corpus Chunks", disable=False):
        	
			corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
			sub_corpus_embeddings =  model.encode(corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, convert_to_tensor=True)
			text_embs.extend(sub_corpus_embeddings.detach().cpu())
		text_embs = torch.stack(text_embs)
	return text_embs
