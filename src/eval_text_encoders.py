import os
import json
import random
import argparse

import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluator import InformationRetrievalEvaluator
from utils.data_utils import load_data, generate_BeIR_resources
from utils.model_utils import get_sentence_transformer_embeddings, get_transformer_embeddings

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

			for model_name in ['all-MiniLM-L12-v2', 'msmarco-bert-base-dot-v5']:
				print('\n', model_name)
				corpus_embeddings = get_sentence_transformer_embeddings(model_name, corpus_content)
				query_embeddings = get_sentence_transformer_embeddings(model_name, queries_content)
				rkey = '{}_{}_seed{}'.format(source, model_name, seed)
				results[rkey] = ire.compute_metrices(corpus_embeddings,query_embeddings)

			for model_name in ['bert-large-cased', 'microsoft/mpnet-base']:
				print('\n', model_name)
				corpus_embeddings = get_transformer_embeddings(model_name, corpus_content)
				query_embeddings = get_transformer_embeddings(model_name, queries_content)
				rkey = '{}_{}_seed{}'.format(source, model_name, seed)
				results[rkey] = ire.compute_metrices(corpus_embeddings,query_embeddings)

			print('\ntf-idf')
			tfidf = TfidfVectorizer()
			corpus_embeddings = tfidf.fit_transform(corpus_content).toarray()
			query_embeddings = tfidf.transform(queries_content).toarray()
			rkey = '{}_{}_seed{}'.format(source, "tf-idf", seed)
			results[rkey] = ire.compute_metrices(corpus_embeddings,query_embeddings)

	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
	with open(args.output_path, 'w+') as _:
			json.dump(results, _)
	print_results(args.output_path)