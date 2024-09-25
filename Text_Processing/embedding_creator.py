# embedding_creator.py

import numpy as np
import os
from gensim.models import Word2Vec
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

class EmbeddingCreator:
    def __init__(
        self,
        embedding_method="glove",
        embedding_dim=100,
        glove_cache_path=None,
        word2vec_model_path=None,
        bert_model_name="bert-base-uncased",
        bert_cache_dir=None,
        device="cuda"
    ):
        self.embedding_method = embedding_method.lower()
        self.embedding_dim = embedding_dim
        self.glove = None
        self.word2vec_model = None
        self.bert_model = None
        self.tokenizer = None

        # Determine device for BERT
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load embeddings based on the specified method
        if self.embedding_method == "glove":
            self._load_glove(glove_cache_path)
        elif self.embedding_method == "word2vec":
            self._load_word2vec(word2vec_model_path)
        elif self.embedding_method == "bert":
            self._load_bert(bert_model_name, bert_cache_dir)
        else:
            raise ValueError("Unsupported embedding method. Choose from 'glove', 'word2vec', or 'bert'.")

    def _load_glove(self, glove_cache_path):
        if not glove_cache_path:
            raise ValueError("glove_cache_path must be provided for GloVe embeddings.")
        self.glove = GloVe(name="6B", dim=self.embedding_dim, cache=glove_cache_path)

    def _load_word2vec(self, word2vec_model_path):
        if not word2vec_model_path or not os.path.exists(word2vec_model_path):
            raise ValueError("A valid word2vec_model_path must be provided for Word2Vec embeddings.")
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        if self.word2vec_model.vector_size != self.embedding_dim:
            raise ValueError(f"Word2Vec model dimension ({self.word2vec_model.vector_size}) does not match embedding_dim ({self.embedding_dim}).")

    def _load_bert(self, bert_model_name, bert_cache_dir):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.embedding_dim = self.bert_model.config.hidden_size  # Update embedding_dim to BERT's hidden size

    def get_embedding(self, tokens):
        if self.embedding_method in ["glove", "word2vec"]:
            return self._get_average_embedding(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_embedding(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def _get_average_embedding(self, tokens):
        embeddings = []
        for token in tokens:
            if self.embedding_method == "glove" and token in self.glove.stoi:
                embeddings.append(self.glove[token].numpy())
            elif self.embedding_method == "word2vec" and token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                continue
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def _get_bert_embedding(self, tokens):
        text = ' '.join(tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding
