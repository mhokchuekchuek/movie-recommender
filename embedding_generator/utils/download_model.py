from sentence_transformers import SentenceTransformer

model = SentenceTransformer("bert-base-nli-mean-tokens")
model.save("src/model")
