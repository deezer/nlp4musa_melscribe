import os
import argparse
import pandas as pd
from typing import List
from utils.data_utils import load_data, generate_BeIR_resources, save_BeIR_resources


def generate_train_data(sources: List[str], input_path: str, output_path: str, random_seed: int = 42, docs_per_query: int = 3) -> None:
	data = {}
	for source in sources:
		print('Extracting from', source, 'all examples')
		data[source] = load_data(source, input_path, split='train')
	df = pd.concat(data.values())
	queries, corpus, train = generate_BeIR_resources(df, random_seed, docs_per_query)
	print(len(queries), 'queries', len(corpus), 'tags')
	# Save BeIR resources
	if not os.path.exists(os.path.join(output_path, 'qgen-qrels')):
		os.makedirs(os.path.join(output_path, 'qgen-qrels'))
	save_BeIR_resources(queries, corpus, train, output_path, for_test=False)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_path', dest='output_path', type=str, required=True, help='Path where to save the BeIR resources')
	parser.add_argument('--sources', nargs='+', choices=['lpms-mtt', 'lpms-mc', 'lpms-msd'], dest='sources', required=True, help='which sources to use for generating BeIR resources')
	parser.add_argument('--input_path', dest='input_path', type=str, required=True, help='Path where to find the source files')
	parser.add_argument('--random_seed', dest='random_seed', type=int, default=42, help='Random seed')
	parser.add_argument('--docs_per_query', dest='docs_per_query', type=int, default=3, help='How many sets of tags per query to sample.')
	args = parser.parse_args()
	generate_train_data(args.sources, args.input_path, args.output_path, args.random_seed, args.docs_per_query)