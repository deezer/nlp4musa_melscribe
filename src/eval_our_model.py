import os
import json
import random
import argparse

import numpy as np

from evaluator import InformationRetrievalEvaluator
from utils.data_utils import load_data, generate_BeIR_resources
from utils.model_utils import get_sentence_transformer_embeddings

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
	parser.add_argument('--our_model_path', dest='our_model_path', type=str, required=True, help='Path where to find our model')
	
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

			print('\n Ours')
			corpus_embeddings = get_sentence_transformer_embeddings(args.our_model_path, corpus_content)
			query_embeddings = get_sentence_transformer_embeddings(args.our_model_path, queries_content)
			rkey = '{}_{}_seed{}'.format(source, 'ours', seed)
			results[rkey] = ire.compute_metrices(corpus_embeddings,query_embeddings)


	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
	with open(args.output_path, 'w+') as _:
			json.dump(results, _)
	print_results(args.output_path)