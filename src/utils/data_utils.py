import os
import uuid
import json
import random
import pandas as pd
import numpy as np
from typing import Any, List, Tuple

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer

SOURCES = {
	'lpms-mtt': 'LP-MusicCaps-MTT/[split].csv',
	'lpms-msd': 'LP-MusicCaps-MSD/[split].csv', 
	'lpms-mc': 'LP-MusicCaps-MC/test.csv',
	'lpms-mc-rephrased': 'LP-MusicCaps-MC/test.csv',
	}

TAG_COLUMN = {
	'lpms-mtt': 'tag_top188',
	'lpms-mc': 'aspect_list', 
	'lpms-msd': 'tag',
	'lpms-mc-rephrased': 'aspect_list',
	}

CAPTION_COLUMN = {
	'lpms-mtt': 'caption_writing',
	'lpms-mc': 'caption_writing', 
	'lpms-msd': 'caption_writing',
	'lpms-mc-rephrased': 'caption_writing', 
}

ID_COLUMN = {
	'lpms-mtt': 'track_id',
	'lpms-mc': 'ytid', 
	'lpms-msd': 'track_id',
	'lpms-mc-rephrased': 'ytid',
}


def load_data(source: str, input_path: str, split: str='test') -> List[Tuple[str,str, str]]:
	path = os.path.join(input_path, SOURCES[source].replace('[split]', split))
	
	print('... reading from', path)
	
	df = pd.read_csv(path)
	tags_col = TAG_COLUMN[source]
	caption_col = CAPTION_COLUMN[source]
	id_col = ID_COLUMN[source]

	# remove lines without caption
	df = df.dropna()
	# format and clean a bit the text
	df[tags_col] = df[tags_col].map(lambda al: al.replace('[', '').replace(']', '').replace("' '", ", ").replace("'", "").replace('\n', ','))
	df[caption_col] = df[caption_col].map(lambda c: c.replace('\n', ' '))
	df = df.rename(columns={caption_col: "caption", tags_col: "tags", id_col: "id"})

	if source in ['lpms-mc-rephrased', 'lpms-mc']:
		rephrased_df = pd.read_csv(os.path.join(input_path, SOURCES[source].replace('test.csv', 'mc_rephrased.csv')))
		rephrased = dict(zip(rephrased_df.caption, rephrased_df.caption_rephrased))
		new_rephrased_col = []
		for caption in df.caption:
			if caption in rephrased:
				new_rephrased_col.append(rephrased[caption])
			else:
				new_rephrased_col.append('')
		df['new_caption'] = new_rephrased_col
		df = df.drop(df[df.new_caption==''].index)
		if source == 'lpms-mc-rephrased':
			df = df.rename(columns={"caption": "original_caption", 'new_caption':'caption'})
	return df


def generate_BeIR_singletag_resources(data) -> (List[Any], List[Any], List[Any]):
	queries = []
	corpus = []
	relevant_tags = [] # mapping between query / caption and corpus tags
	tags_to_ids = {}
	data_records = data.to_dict(orient='records')
	for i in range(len(data_records)):
		caption = data_records[i]['caption']
		query_id = str(uuid.uuid4())
		queries.append((query_id, caption))
		tags = set(data_records[i]['tags'].split(', '))
		relevant_tags_per_query = []
		for t in tags:
			if t in tags_to_ids:
				doc_id = tags_to_ids[t]
			else:
				doc_id = str(uuid.uuid4())
				tags_to_ids[t] = doc_id
				corpus.append((doc_id, t))
			relevant_tags_per_query.append(doc_id)
		relevant_tags.append((query_id, set(relevant_tags_per_query)))
	return queries, corpus, relevant_tags


def generate_BeIR_resources(df: pd.DataFrame, random_seed: int = 42, docs_per_query: int = 1) -> (List[Any], List[Any], List[Any]):
	random.seed(random_seed)
	
	queries = []
	corpus = []
	relevant_tags = [] # mapping between query / caption and corpus tags
	tags_to_ids = {}

	data = df[["id", "tags", "caption"]].to_dict(orient='records')
	for i in range(len(data)):
		track_id = data[i]['id']
		tags = data[i]['tags'].split(', ')
		caption = data[i]['caption']

		# split in sentences
		sentences = caption.split('. ')

		# get overlapping and non-overlapping tags with each sentence
		# simply based on exact string match for now
		# another option would have been to use cross-encoders
		tags_per_sent = {}
		for sent in sentences:
			tags_per_sent[sent] = {}
			tags_per_sent[sent]['ematch'] = []
			tags_per_sent[sent]['others'] = []
			for tag in tags:
				if tag in sent:
					tags_per_sent[sent]['ematch'].append(tag)
				else:
					tags_per_sent[sent]['others'].append(tag)
	
		# we generate multiple queries per track
		for sent in sentences:
			query_id = str(track_id) + '_' + str(uuid.uuid4())
			queries.append((query_id, sent))
		
			# we sample multiple sets tags for each query
			# in order to ensure that we have multiple documents per query
			docs = set()
			min_len = min(len(tags_per_sent[sent]['ematch']), len(tags_per_sent[sent]['others']))
			for i in range(docs_per_query):
				sampled_tag_list = []
				# first sample from the exact match tags
				if len(tags_per_sent[sent]['ematch']) > 0:
					k1 = random.choice(range(1, len(tags_per_sent[sent]['ematch']) + 1))
					sampled_tag_list.extend(random.sample(tags_per_sent[sent]['ematch'], k1))

				# then sample from the other tags
				if len(tags_per_sent[sent]['others']) > 0:
					k2 = random.choice(range(0, min_len + 1))
					sampled_tag_list.extend(random.sample(tags_per_sent[sent]['others'], k2))
				# shuffle the tags and add them to the list of documents
				random.shuffle(sampled_tag_list)
				docs.add(', '.join(sampled_tag_list))

			doc_ids = set()
			for doc in docs:
				if doc in tags_to_ids:
					doc_id = tags_to_ids[doc]
				else:
					doc_id = str(uuid.uuid4())
					tags_to_ids[doc] = doc_id
					corpus.append((doc_id, doc))
				doc_ids.add(doc_id)
			relevant_tags.append((query_id, doc_ids))

	return queries, corpus, relevant_tags


def save_BeIR_resources(queries, corpus, relevant_tags, output_path, for_test=False):
	BeIR_queries = []
	for query_id, caption in queries:
		# Create the query entry in BeIR format
		query = {}
		query["_id"] = query_id
		query["metadata"] = {}
		query["text"] = caption
		BeIR_queries.append(query)

	BeIR_corpus = []
	for doc_id, tags in corpus:
		# Create the doc entry in the BeIR format	
		doc = {}
		doc["_id"] = doc_id
		doc["title"] = ""
		doc["text"] = tags
		BeIR_corpus.append(doc)

	mappings = []
	for query_id, doc_ids in relevant_tags:
		for doc_id in doc_ids:
			mappings.append((query_id, doc_id, 1))

	if for_test:
		rel_tags_file_name = 'qrels/test.tsv'
		queries_file_name = 'queries.jsonl'
	else:
		rel_tags_file_name = 'qgen-qrels/train.tsv'
		queries_file_name = 'qgen-queries.jsonl'

	with open(os.path.join(output_path, "corpus.jsonl"), "w") as _:
		for c in BeIR_corpus:
			json.dump(c, _)
			_.write('\n')

	with open(os.path.join(output_path, queries_file_name), "w") as _:
		for q in BeIR_queries:
			json.dump(q, _)
			_.write('\n')

	out_df = pd.DataFrame(mappings, columns=["query-id", "corpus-id", "score"])
	out_df.to_csv(os.path.join(output_path, rel_tags_file_name), sep='\t', index=False)
