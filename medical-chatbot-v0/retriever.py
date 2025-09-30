from FlagEmbedding import FlagModel
import faiss
import numpy as np
import datasets
from datasets import load_dataset
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
import pandas as pd


logger = logging.getLogger(__name__)

class RetrievalResult(BaseModel):
    """
    A class to represent the result of a retrieval operation.
    """
    doc_id: Any = Field(None, description="The ID of the retrieved document.")
    question: Any = Field(None, description="The question used for retrieval.")
    score: Any = Field(None, description="The score of the retrieved document.")
    data: Any = Field(None, description="The data of the retrieved document.")

    def set_sorted_indices(self, sorted_indices: List[int]):
        """
        Set the sorted indices for the retrieval result.
        Args:
            sorted_indices (List[int]): The sorted indices of the retrieved documents.
        """
        self.doc_id = [self.doc_id[i] for i in sorted_indices]
        self.score = [self.score[i] for i in sorted_indices]
        self.data = self.data.iloc[sorted_indices]

    def __getitem__(self, index):
        """
        Get the item at the specified index.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            RetrievalResult: The item at the specified index.
        """
        return {
            "doc_id": self.doc_id[index],
            "question": self.question,
            "score": self.score[index],
            "series": self.data.iloc[index],
        }

    def __len__(self):
        """
        Get the length of the result.
        Returns:
            int: The length of the result.
        """
        return len(self.doc_id)

class RagRetriever:
    """
    RagRetriever is a class that retrieves documents from a dataset using a retriever model.
    It is used in conjunction with the RAG (Retrieval-Augmented Generation) model.
    """

    def __init__(self, model, dataset=None, k=3):
        
        if isinstance(model, str):
            self.embedding_model = FlagModel(model)
        else:
            self.embedding_model = model
        self.dataset : datasets.DatasetDict = dataset
        self.topk = k
        
    def _retrieve(self, query, k):
        """
        Retrieve documents from the dataset using the retriever model.
        Args:
            query (str): The query string to retrieve documents for.
        Returns:
            List[dict]: A list of retrieved documents.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not set. Please provide a dataset.")
        if not k:
            k = self.topk
        
        # Tokenize the query
        logger.info(f"Retrieving documents for query: {query}")
        
        # Get the embeddings for the query
        q_embedding = self.embedding_model.encode_queries([query], convert_to_numpy=True)

        
        # Perform retrieval using the embeddings
        scores, idxs = self.dataset.get_index("embeddings").search(q_embedding, k)

        # Retrieve the documents based on the indices
        ret = self.dataset[idxs]

        df = pd.DataFrame(ret)

        logger.debug(f"Retrieved documents: {df}")

        for x in ret:
            logger.debug(f"Retrieved document: {x}")
            
        return RetrievalResult(
            doc_id=idxs,
            score=scores,
            question=query,
            data=df,
        )

    def retrieve(self, questions: List[str], n_docs: int) -> List[RetrievalResult]:
        ret_batched = []
        for que in questions:
            start_time = time.time()
            item = self._retrieve(que, n_docs)
            logger.debug(
                f"index search time: {time.time() - start_time} sec, batch size {len(questions)}"
            )
            ret_batched.append(item)
        return ret_batched
