import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from FlagEmbedding import FlagModel, FlagAutoModel
from retriever import RagRetriever, RetrievalResult
import faiss
import torch
from qwen_embedding import EmbeddingModel
import json
import hashlib 
from datasets import Features, Sequence, Value, load_dataset, Dataset, load_from_disk
import openai
from openai import AzureOpenAI
from reranker import RagReranker

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagSequenceForGeneration,
    RagTokenizer,
)


logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict, column_names: List[str]) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: dict, encoder: FlagModel, column_names: List[str]) -> dict:
    """Compute the DPR embeddings of document passages"""
    # column_datas = [ documents[col] for col in column_names ]
    corpus = []
    # Concatenate the columns into a single string
    for i in range(len(documents[column_names[0]])):
        text = []
        for col in column_names:
            if not documents[col][i]: continue
            stext = documents[col][i].split()
            text.append(" ".join(stext))
        corpus.append(" ".join(text))
    # corpus = [ f"{title} {text}" for title, text in zip(documents["title"], documents["text"]) ]
    embeddings = encoder.encode(corpus, convert_to_numpy=True)
    return {"embeddings": embeddings}



def database_processing(
    database_file: "Dataset",
    encoder: "FlagModel",
    save_dir: str,
    column_names: str = "title,text",
    num_proc: int = 4,
    batch_size: int = 8,
):
    """
    Process the database file to create a dataset with embeddings.
    Args:
        database_file (str): Path to the database file.
        encoder (DPRContextEncoder): The encoder to use for embedding.
    Returns:
        Dataset: The processed dataset with embeddings.
    """

    logger.info("Step 1 - Create the dataset")
    column_names = column_names.split(",")

    if database_file.endswith(".csv"):
        dataset = load_dataset("csv", data_files=database_file, split="train")
    elif database_file.endswith(".json"):
        dataset = load_dataset("json", data_files=database_file, split="train")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    
    # Split the documents into passages
    # dataset = dataset.map(partial(split_documents, column_names=column_names), batched=True, num_proc=num_proc)

    logger.info("Set column names")
    logger.info(f"Column names: {column_names}")

    columns = {
        col: Value("string")
        for col in column_names
    }
    columns["embeddings"] = Sequence(Value("float32"))

    # new_features = Features(columns) 
    # Compute the embeddings
    dataset = dataset.map(
        partial(embed, encoder=encoder, column_names=column_names),
        batched=True,
        batch_size=batch_size,
        # features=new_features,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset.save_to_disk(os.path.join(save_dir, "DB_dataset"))

    logger.info("Step 2 - Index the dataset")
    index = faiss.IndexHNSWFlat(encoder.model.config.hidden_size, encoder.model.config.hidden_size, faiss.METRIC_INNER_PRODUCT)

    dataset.add_faiss_index("embeddings", custom_index=index)
    # And save the index
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset.get_index("embeddings").save(os.path.join(save_dir, "DB_index.faiss"))
    # md5_hash = hashlib.md5()

    # md5_hash.update(database_file + " ".join(column_names))
    # result = md5_hash.hexdigest()

    with open(os.path.join(save_dir, "DB_index.info"), "w") as f:
        f.write(json.dumps({
                "column_names": column_names,
                "database_file": database_file,
        }))

    return dataset


def load_dataset_from_disk(path: str) -> Dataset:
    """
    Load a dataset from disk.
    Args:
        path (str): Path to the dataset directory.
    Returns:
        Dataset: The loaded dataset.
    """
    return load_from_disk(path)

def load_index_from_disk(path: str) -> faiss.Index:
    """
    Load an index from disk.
    Args:
        path (str): Path to the index file.
    Returns:
        faiss.Index: The loaded index.
    """
    return faiss.read_index(path)


def load_dataset_and_index(
    dataset_dir: str,
) -> Dataset:
    """
    Load a dataset and its index from disk.
    Args:

        dataset_path (str): Path to the dataset directory.
        index_path (str): Path to the index file.
    Returns:
        Dataset: The loaded dataset with the index.
    """
    logger.info("Loading dataset and index from disk")
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset path {dataset_dir} does not exist.")
    dataset_path = os.path.join(dataset_dir, "DB_dataset")
    index_path = os.path.join(dataset_dir, "DB_index.faiss")
    dataset = load_dataset_from_disk(dataset_path)
    dataset.load_faiss_index("embeddings", file = os.path.join(dataset_dir, "DB_index.faiss"))
    return dataset




def main(
    rag_args: "RagArguments",
    processing_args: "ProcessingArguments",
):
    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(rag_args.db_path)

    encoder = FlagAutoModel.from_finetuned(rag_args.retrive_model_name_or_path, devices=device)

    # You can load a Dataset object this way

    if rag_args.dataset_cache_dir and os.path.exists(rag_args.dataset_cache_dir):
        dataset = load_dataset_and_index(rag_args.dataset_save_dir)
    else:
        dataset = database_processing(
            database_file = rag_args.db_path,
            encoder = encoder,
            save_dir = rag_args.dataset_save_dir,
            column_names = rag_args.column_names,
            num_proc = processing_args.num_proc,
        )
    retriever = RagRetriever(
        model = encoder,
        dataset = dataset,
    )
    questions = []
    if rag_args.question:
        questions.append(rag_args.question)
    else:
        with open(rag_args.input_file, "r") as f:
            questions = [line.strip() for line in f.readlines()]

    while True:
        questions = []
        que = input("Please enter a question (or 'exit' to quit): ")
        if que.lower() == "exit":
            break
        questions.append(que)

        results = retriever.retrieve(questions, n_docs=3)
        for result in results:
            logger.info(f"Question: {result.question}\n")
            for x in range(len(result)):
                doc = result[x]
                series = result[x]['series']
                logger.info(f"Retrieved Document: ")
                logger.info(f"{series}")
                logger.info(f"Score: {doc['score']}\n")

class RAG:

    def __init__(self, 
            corpus_path: str,
            retrive_model_name_or_path: str,
            reranker_model_name_or_path: str,
            generate_model_name: str,
            base_url: str,
            api_key: str,
            context_columns: List[str] = ["title", "text"],
            template: str = "Please answer the question based on the context: {context} Question: {question}",
            system_prompt_template: Optional[str] = None,
            user_prompt_template: Optional[str] = None,
            top_k: int = 3,
            device: str = "cuda",
            azure_openai_args = None,
            use_azure = False,
            column_names: str = "title,text",
        ):
        self.corpus_path = corpus_path
        self.retrive_model_name_or_path = retrive_model_name_or_path
        self.reranker_model_name_or_path = reranker_model_name_or_path
        self.generate_model_name = generate_model_name
        self.base_url = base_url
        self.api_key = api_key
        self.top_k = top_k
        self.device = device
        self.template = template
        self.system_prompt_template = system_prompt_template or (
            "You are a retrieval-augmented assistant. Use only the passages in the knowledge base context below.\n\nContext:\n{context}"
        )
        self.user_prompt_template = user_prompt_template or "{question}"
        self.context_columns = context_columns

        if "Qwen3" in retrive_model_name_or_path:
            self.encoder = EmbeddingModel(embedding_model_path=retrive_model_name_or_path, device=device)
        else:
            self.encoder = FlagAutoModel.from_finetuned(retrive_model_name_or_path, devices=device)

        if os.path.exists(os.path.join(self.corpus_path, "DB_index.info")):
            self.dataset = load_dataset_and_index(self.corpus_path)
        else:
            # dataset_save_dir = str(Path(self.corpus_path).parent / "tmp" )
            # dataset_save_dir = str(Path(self.corpus_path).parent / "tmp_qwen8b" )
            dataset_save_dir = str(Path(self.corpus_path).parent / "tmp_bge" )
            self.dataset = database_processing(
                database_file = self.corpus_path,
                encoder = self.encoder,
                save_dir = dataset_save_dir,
                column_names = column_names,
            )

        self.retriever = RagRetriever(
            model = self.encoder,
            dataset = self.dataset,
        )

        self.reranker = RagReranker(
            reranker_model_path = self.reranker_model_name_or_path,
            device = device
        )

        if use_azure and azure_openai_args:
            self.client = AzureOpenAI(**azure_openai_args)
        else:
            self.client = openai.Client(api_key=self.api_key, base_url=base_url)


    def retrieve(self, questions: List[str], n_docs: int=None) -> List[RetrievalResult]:
        if n_docs is None:
            n_docs = self.top_k
        
        ret = self.retriever.retrieve(questions, n_docs)
        return ret

    def generate(self, question: str="", n_docs: int=None) -> List[str]:
        if n_docs is None:
            n_docs = self.top_k

        context = []

        logger.info(f"Question: {question}")
        
        ret = self.retriever._retrieve(question, n_docs)

        for x in range(len(ret)):
            doc = ret[x]
            series = ret[x]['series']
            scontent = []
            for col in self.context_columns:
                if series[col]:
                    scontent.append( series[col] )
                logger.info(f"Retrieved Document: ")
                logger.info(f"{series}")
                logger.info(f"Score: {doc['score']}\n")
            context.append(" ".join(scontent))

        sorted_documents, sorted_scores, sorted_indices = self.reranker.sort_one_query(question, context)
        context_block = "\n\n".join(sorted_documents)
        system_message = self.system_prompt_template.format(context=context_block)
        user_message = self.user_prompt_template.format(question=question)
        ret.set_sorted_indices(sorted_indices)

        logger.info("System prompt prepared for generation.")
        logger.debug(system_message)
        logger.info("User prompt prepared for generation.")
        logger.debug(user_message)

        answer = self._chat_completion([
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ])
        return {
            "response": answer,
            "documents": ret
        }

    def generate_raw(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, str]:
        """Send the provided prompt directly to the generator model without RAG context."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        answer = self._chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        return {"response": answer}

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Wrapper around the chat completion call to keep arguments consistent."""
        response = self.client.chat.completions.create(
            model=self.generate_model_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
        logger.debug(response)
        return response.choices[0].message.content


@dataclass
class RagArguments:
    db_path: str = field(
        default=str(Path(__file__).parent / "test_data" / "my_knowledge_dataset.csv"),
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default=None,
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    retrive_model_name_or_path: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={"help": "The retrive model to use. e.g. 'BAAI/bge-base-en-v1.5'"},
    )
    generate_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        metadata={
            "help": (
                "The generate model to use. e.g. 'Qwen/Qwen2.5-VL-72B-Instruct'"
            )
        },
    )
    column_names: str = field(
        default="title,text",
        metadata={"help": "Column names to use for the dataset. Default is 'title,text'."},
    )
    input_file: Optional[str] = field(
        default=str(Path(__file__).parent / "ques" / "ques.txt"),
        metadata={ "help": "Path to a file with questions to ask the model"}
    )
    output_dir: Optional[str] = field(
        default=str(Path(__file__).parent / "output" ),
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )
    dataset_save_dir: Optional[str] = field(
        default=str(Path(__file__).parent / "tmp" ),
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        # default=str(Path(__file__).parent / "tmp" ),
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )




if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagArguments, ProcessingArguments))
    rag_example_args, processing_args = parser.parse_args_into_dataclasses()
    main(rag_example_args, processing_args)
