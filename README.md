# nlp4musa_melscribe

This repository provides Python code to reproduce the experiments from the article [**Harnessing High-Level Song Descriptors towards Natural Language-Based Music Recommendation**](https://arxiv.org/abs/2411.05649), accepted for publication to [**NLP4MusA 2024**](https://sites.google.com/view/nlp4musa-2024/home).

For a summary of this project, please consult the [poster](https://github.com/deezer/nlp4musa_melscribe/blob/main/presentation/poster.pdf) or [slides](https://github.com/deezer/nlp4musa_melscribe/blob/main/presentation/slides.pdf).


## Setup

```sh
git clone https://github.com/deezer/nlp4musa_melscribe.git
cd nlp4musa_melscribe
```

Install the requirements:

```bash
pip install -r requirements.txt
```

**LP-MusicCaps** datasets are available for download on Hugging Face ([MC](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MC), [MTT](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MTT), [MSD](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MSD)). 
Each of these datasets should be read and exported to csv files for each split as we show below for **LP-MusicCaps-MTT**:
```python
from datasets import load_dataset

ds = load_dataset("seungheondoh/LP-MusicCaps-MTT")
ds['test'].to_csv('data/LP-MusicCaps-MTT/test.csv')
ds['train'].to_csv('data/LP-MusicCaps-MTT/train.csv')
ds['valid'].to_csv('data/LP-MusicCaps-MTT/valid.csv')
```
**LP-MusicCaps-MSD** is a gated dataset so you must be authenticated to access it.

Download the fine-tuned models (the cross-encoder and the bi-encoder) from [Zenodo](https://zenodo.org/records/14289764):
```bash
wget https://zenodo.org/records/14289764/files/models.zip
unzip models.zip -d models/
```

## Reproduce paper results

###  Evaluate our model
```bash
python src/eval_our_model.py --output_path results/results_our_model.json --sources  lpms-mtt lpms-msd lpms-mc lpms-mc-rephrased --input_path data/ --our_model_path models/bi-encoder-lpmusicaps-msmarco-bert-base-dot-v5/
```

###  Evaluate text encoder baselines
```bash
python src/eval_text_encoders.py --output_path results/results_text_encoders.json --sources  lpms-mtt lpms-msd lpms-mc lpms-mc-rephrased --input_path data/
```

###  Evaluate text encoder baselines from multimodal models

Set up the baselines (*depending on the environment, `pip install -e src/music-text-representation/` throws an exception regarding the package `sklearn`; as suggested, this should be replaced with `scikit-learn` in the file `setup.py`*):

```bash
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt -P src/laion-clap/
git clone https://github.com/seungheondoh/music-text-representation.git src/music-text-representation/
pip install -e src/music-text-representation/
wget https://zenodo.org/record/7322135/files/mtr.tar.gz -P src/music-text-representation/
tar -zxvf src/music-text-representation/mtr.tar.gz -C src/music-text-representation/
```


Run the evaluation script:
```bash
python src/eval_text_encoder_multimodal_models.py --output_path results/results_text_encoders_multimodal.json --sources  lpms-mtt lpms-msd lpms-mc lpms-mc-rephrased --input_path data/ --ttmr_model_path src/music-text-representation/mtr/ --clap_model_path src/laion-clap/music_audioset_epoch_15_esc_90.14.pt
```

## Fine-tune a model from scratch

Generate training data:
```bash
python src/generate_train_data.py --input_path data/ --output_path data/training_gpl --sources lpms-mtt lpms-msd --random_seed=42 --docs_per_query=3
```

Train the model with the Generative Pseudo-labeling method (GPL):
```bash
python -m  gpl.train  --path_to_generated_data "data/training_gpl"    --base_ckpt "msmarco-bert-base-dot-v5"     --gpl_score_function "cos_sim"     --batch_size_gpl 4   --gpl_steps 140000   --output_dir "models/nlp4musa_seed42"    --retrievers "msmarco-distilbert-base-v3" "msmarco-MiniLM-L-6-v3"     --retriever_score_functions "cos_sim"  --negatives_per_query 30  --cross_encoder "models/cross-encoder-musiccaps-ms-marco-MiniLM-L-6-v2/"    --qgen_prefix "qgen" --max_seq_length 512
```

As described in the paper, we fine-tuned a domain-specific cross-encoder using human-annotated data from the MusicCaps dataset, specifically the model `models/cross-encoder-musiccaps-ms-marco-MiniLM-L-6-v2`. The cross-encoder predicts a similarity score between a music-related longer text (e.g., song descriptions or user requests) and a music descriptor (e.g., tags). This model serves as a teacher to generate soft labels for the training data, which are then used to train the bi-encoder.

## Paper

Please cite our paper if you use this data or code in your work:
```
@InProceedings{Epure2024Harnessing,
 	title={Harnessing High-Level Song Descriptors towards Natural Language-Based Music Recommendation},
  	author={Epure, Elena V. and Meseguer-Brocal, Gabriel and Afchar, Darius and Hennequin, Romain},
  	booktitle={Proceedings of the 3rd Workshop on NLP for Music and Audio (NLP4MusA2024)},
  	month={November},
  	year={2024},
  	publisher = {Association for Computational Linguistics},
}
```
