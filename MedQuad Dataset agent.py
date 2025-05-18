
"""
LLM-based Agent for the MedQuad Dataset

MedQuad is an English medical question-answering dataset designed for AI and NLP research.
It contains QA pairs focused on health topics sourced from reputable institutions like the NIH.
"""

# === Install dependencies ===
# Run in terminal or Colab:
# !pip install llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface bitsandbytes kagglehub

# === Import Libraries ===
import os
import kagglehub
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from llama_index.core import Settings, Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# === Download dataset ===
path = kagglehub.dataset_download(handle="googleai/dataset-metadata-for-cord19")
print("Path to dataset files:", path)

filename_with_path = path + "/" + os.listdir(path)[0]
df_meta_cord19 = pd.read_csv(filename_with_path)

# Filter only rows with non-null descriptions
df_meta_cord19_filtered = df_meta_cord19[df_meta_cord19['description'].notnull()]

# === Create vector store using LlamaIndex ===
VECTOR_STORE_DIR = "./vector_store"

if not os.path.exists(VECTOR_STORE_DIR):
    documents = [Document(text=row['description']) for index, row in tqdm(df_meta_cord19_filtered.iterrows())]
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    print(f"Vector store created and saved to {VECTOR_STORE_DIR}")
else:
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
    index = load_index_from_storage(storage_context)
    print(f"Vector store loaded from {VECTOR_STORE_DIR}")

# === Create LLM Agent ===
llm = HuggingFaceLLM(
    model_name="colesmcintosh/Llama-3.2-1B-Instruct-Mango",
    tokenizer_name="colesmcintosh/Llama-3.2-1B-Instruct-Mango",
    context_window=2048,
    max_new_tokens=256,
    device_map="cuda:0",
    generate_kwargs={"temperature": 0.95, "do_sample": True},
)
Settings.llm = llm

# === Create Chat Engine ===
chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt="You are a medical chatbot. You only answer based on the MedQuad dataset."
)

# === Continuous Chat Loop ===
while True:
    query = input("> ")
    if query.lower() == "quit":
        break
    print("Agent: ", end="", flush=True)
    response = chat_engine.stream_chat(query)
    for token in response.response_gen:
        print(token, end="", flush=True)
    print()

chat_engine.reset()
