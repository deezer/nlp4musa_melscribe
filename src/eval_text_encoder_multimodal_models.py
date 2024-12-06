import os
import json
import random
import argparse

import torch
import numpy as np
from tqdm import trange

from evaluator import InformationRetrievalEvaluator
from utils.data_utils import load_data, generate_BeIR_resources

from utils.ttmr import get_model
from utils.clap import CLAP_Module

def get_clap_embeddings(model_path, corpus):
	corpus_chunk_size = 4
	model = CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
	model.load_ckpt(model_path)

	text_embs = []
	if len(corpus) < corpus_chunk_size:
		text_embs = model.get_text_embedding(corpus, use_tensor=True).detach().cpu()
	else:
		for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc="Corpus Chunks", disable=False):
			
			corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
			sub_corpus_embeddings =  model.get_text_embedding(corpus[corpus_start_idx:corpus_end_idx], use_tensor=True)
			text_embs.extend(sub_corpus_embeddings.detach().cpu())
		text_embs = torch.stack(text_embs)
	return text_embs


def get_ttmr_embeddings(model_path, corpus, framework='contrastive', text_type='bert', text_rep='stochastic'):
	corpus_chunk_size = 100	
	model, tokenizer, config = get_model(framework=framework, text_type=text_type, text_rep=text_rep, path_prefix=model_path)
	# model  = model.to(device)

	text_embs = []
	if len(corpus) < corpus_chunk_size:
		with torch.no_grad():
			text_input = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True)['input_ids'].to('cpu')
			text_embs = model.encode_bert_text(text_input, None).detach().cpu()
	else:
		for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc="Corpus Chunks", disable=False):
			
			corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
			with torch.no_grad():
				text_input = tokenizer(corpus[corpus_start_idx:corpus_end_idx], return_tensors="pt", padding=True, truncation=True)['input_ids'].to('cpu')
				sub_corpus_embeddings =  model.encode_bert_text(text_input[0:corpus_chunk_size], None).detach().cpu()
				text_embs.extend(sub_corpus_embeddings)
		text_embs = torch.stack(text_embs)
	return text_embs


def print_results(file_path):
	results = json.load(open(file_path))
	results_aggregated = {}
	for k in results:
		source, model, seed = k.split('_')
		#map_rkey = 'map_{}_{}'.format(source, model)
		recall_rkey = 'recall_{}_{}'.format(source, model)
		#if map_rkey not in results_aggregated:
		#	results_aggregated[map_rkey] = []
		if recall_rkey not in results_aggregated:
			results_aggregated[recall_rkey] = []
		#results_aggregated[map_rkey].append(results[k]['cos_sim']['map@k']['100'])
		results_aggregated[recall_rkey].append(results[k]['cos_sim']['recall@k']['10'])
	for k in results_aggregated:
		print(k, "{:.3f}".format(np.mean(results_aggregated[k])), "{:.3f}".format(np.std(results_aggregated[k])))


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_path', dest='output_path', type=str, required=True, help='Path where to save the results')
	parser.add_argument('--sources', nargs='+', choices=['lpms-mtt', 'lpms-msd', 'lpms-mc', 'lpms-mc-rephrased'], dest='sources', required=True, help='which sources to use for generating BeIR resources')
	parser.add_argument('--input_path', dest='input_path', type=str, required=True, help='Path where to find the source files')
	parser.add_argument('--ttmr_model_path', dest='ttmr_model_path', type=str, required=True, help='Path where to find the model') # e.g.: src/music-text-representation/mtr/
	parser.add_argument('--clap_model_path', dest='clap_model_path', type=str, required=True, help='Path where to find the model') # e.g.: src/laion-clap/music_audioset_epoch_15_esc_90.14.pt
	
	args = parser.parse_args()

	results = {}
	for source in args.sources:
		for seed in range(3):
			print('\n', source)
			data = load_data(source, args.input_path, split='test')
			queries, corpus, relevant_tags = generate_BeIR_resources(
				data, random_seed=seed, docs_per_query=1)
				
			queries = dict(queries)
			corpus = dict(corpus)
			relevant_tags = dict(relevant_tags)
				
			queries_ids = []
			for qid in queries:
				if qid in relevant_tags and len(relevant_tags[qid]) > 0:
					queries_ids.append(qid)
			queries_content = [queries[qid] for qid in queries_ids]
				
			corpus_ids = list(corpus.keys())
			corpus_content = [corpus[cid] for cid in corpus_ids]

			ire = InformationRetrievalEvaluator(queries, corpus, relevant_tags)

			print('\nClap')
			corpus_embeddings = get_clap_embeddings(args.clap_model_path, corpus_content)
			query_embeddings = get_clap_embeddings(args.clap_model_path, queries_content)
			rkey = '{}_{}_seed{}'.format(source, "clap", seed)
			results[rkey] = ire.compute_metrices(corpus_embeddings, query_embeddings)

			print('\nttmr')
			corpus_embeddings = get_ttmr_embeddings(args.ttmr_model_path, corpus_content)
			query_embeddings = get_ttmr_embeddings(args.ttmr_model_path, queries_content)
			rkey = '{}_{}_seed{}'.format(source, "ttmr", seed)
			results[rkey] = ire.compute_metrices(corpus_embeddings, query_embeddings)

	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
	with open(args.output_path, 'w+') as _:
			json.dump(results, _)
	print_results(args.output_path)