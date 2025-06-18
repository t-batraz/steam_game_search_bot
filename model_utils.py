import torch
import numpy as np
import faiss
import json
import gc
from sentence_transformers import SentenceTransformer
from mxbai_rerank import MxbaiRerankV2
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelManager:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.embed_model = None
        self.index = None
        self.rerank = None
        self.review_tokenizer = None
        self.review_model = None

    def load_models(self):
        self.embed_model = SentenceTransformer(
            'jinaai/jina-embeddings-v3',
            trust_remote_code=True
        ).to(self.device)

        self.index = faiss.read_index(self.config['faiss_index_path'])

        self.rerank = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )

        self.review_tokenizer = AutoTokenizer.from_pretrained(
            self.config['review_model_name']
        )
        self.review_model = AutoModelForCausalLM.from_pretrained(
            self.config['review_model_name'],
            device_map="auto",
            torch_dtype="auto",
            quantization_config=bnb_config,
        )
        return self

    def search_games(self, query, top_k=10):
        embed_vec = self.embed_model.encode([query])
        D, I = self.index.search(np.array(embed_vec, dtype='float32'), k=top_k)
        return [int(i) for i in I[0]]

    def load_game_data(self, game_id):
        file_path = f"{self.config['data_dir']}/{game_id}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def rerank_games(self, query, games_data, top_k=3):
        game_texts = [
            f"{game['name']}\n{game['detailed_description']}\n"
            + '\n'.join(game['users_reviews'])
            for game in games_data
        ]
        reranked = self.rerank.rank(
            query,
            game_texts,
            return_documents=False,
            top_k=top_k
        )
        return [games_data[r.index] for r in reranked]

    def generate_review(self, game_data):
        messages = [
            {
                "role": "system",
                "content": "Below are reviews from users. Analyze them and write a short, professional review of the game."
            },
            {"role": "user", "content": f"Game name: {game_data['name']}"},
            {
                "role": "user",
                "content": '\n'.join([r.replace('\n', ' ') for r in game_data['users_reviews'][:5]])
            }
        ]

        input_ids = self.review_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.review_model.generate(input_ids, max_new_tokens=256)
        review = self.review_tokenizer.decode(
            outputs[0][input_ids.size(1):],
            skip_special_tokens=True
        )
        return review

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()