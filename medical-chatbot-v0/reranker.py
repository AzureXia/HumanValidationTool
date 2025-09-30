# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class RagReranker:

    def __init__(self, reranker_model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(reranker_model_path).eval()
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores


    def rerank(self, queries, documents, instruction=None):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        pairs = [self.format_instruction(instruction or task, query, doc) for query, doc in zip(queries, documents)]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        return scores
    
    def _sort_documents(self, queries, documents, instruction=None):
        scores = self.rerank(queries, documents, instruction)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_documents = [documents[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        return sorted_documents, sorted_scores, sorted_indices

    def sort_one_query(self, query, documents, instruction=None):
        """
        Sort documents for a single query.
        """
        return self._sort_documents([query] * len(documents) , documents, instruction)

if __name__ == "__main__":
    reranker_model_path = '/root/data/rag/final/Qwen3-Reranker-0.6B'

    reranker = RagReranker(reranker_model_path=reranker_model_path, device='cuda')
    queries = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."]
    sorted_documents, sorted_scores = reranker.sort_one_query(queries, documents)
    print(sorted_documents)
    print(sorted_scores)

