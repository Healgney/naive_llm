import pandas as pd
import torch
import json, yaml
import os, re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
from peft import PeftModel, PeftConfig
from bert_score import score as bert_score
from sklearn.metrics import f1_score

from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder

from doc_process import DocProcessor
from judges import Judges
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #for gpu-calculation debugging

import os
from pathlib import Path

def is_valid_path(path_str):
    try:
        Path(path_str).resolve(strict=False)
        return True
    except (OSError, ValueError):
        return False

class Retriever:
    def __init__(self, chunks, embedding_model_path:str, reranker_model_path:str):
        if not isinstance(embedding_model_path, str) and not isinstance(reranker_model_path, str):
            raise ValueError("embedding_model_path and reranker_model_path must be str")

        match = re.search(r'models--([^/]+)--([^/]+)', embedding_model_path)
        if match:
            embedding_model = f"{match.group(1)}/{match.group(2)}"
        else:
            embedding_model = embedding_model_path #exactly the model name

        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(
                model_name=embedding_model_path,
            ),
            persist_directory=f".data/vec_db/vector_db_{embedding_model}" #gotta store the VecDB differently
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=5
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.3, 0.7]
        )

        self.reranker = CrossEncoder(
            model_name_or_path=reranker_model_path,
            device="cuda" if torch.has_cuda else "cpu"
        )

    def retrieve(self, query, top_k=3):
        docs = self.ensemble_retriever.get_relevant_documents(query)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:top_k]]


class LLM_w_RAG:
    def __init__(self,
                 doc_dir:str,
                 model_name_or_path:str,
                 peft_model_path:str,
                 embedding_model_path:str,
                 reranker_model_path:str,
                 cache_dir=None
                 ):
        self.model_name_or_path = model_name_or_path
        self.peft_model_path = peft_model_path

        processor = DocProcessor(data_path=doc_dir, embd_model=embedding_model_path)
        chunks = processor.process_documents()

        self.retriever = Retriever(chunks, embedding_model_path, reranker_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                       trust_remote_code=True,
                                                       use_fast=False
                                                       )
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)

        self.model = self.model.merge_and_unload() # Essential, keep original Model class

        self.model.eval()

    def generate_prompt(self, question, contexts):
        context_str = "\n\n".join([
            f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
            for doc in contexts
        ])
        # instruction = (
        #         "你是一个《周易》研究专家，善于结合《周易》原文、象辞、传统卦象解读为用户解答卜卦、卦义与哲理相关的问题。请基于以下知识内容，准确而简洁地作答：\n\n"
        #         f"{context_str}\n{'#' * 7}" + "\n\n"
        #                    "回答要求：\n"
        #                     "- 答案必须基于上方检索内容，不要编造。\n"
        #                     "- 如有卦名、爻辞或象辞原文，请适当引用。\n"
        #                     "- 若内容不足以回答问题，可明确说明「根据已有资料，无法确定具体答案」。\n"
        #                     "- 风格应尊重《周易》的哲理风范，表达简明、有深意、切中主题。\n"
        #                    "请按照上述要求回答以下问题："
        # )

        instruction = (
                "你是一位《周易》领域的专家，擅长结合原文、象辞和传统理解来解答卦象、爻义及相关哲理问题。\n\n"
                f"{context_str}\n{'#' * 7}" + "\n\n"
                                              "请根据以上知识内容，回答下列用户问题。\n"
                                              "评判标准以 **回答内容的完整性和准确性为主**，无需过于关注语言风格或表达方式。\n\n"
                                              "回答建议：\n"
                                              "- 尽量基于上述内容作答，如有卦名、爻辞或象辞原文可以适当引用。\n"
                                              "- 若资料不足，可合理推断或简要说明“不确定”。\n"
                                              "- 答案风格可简明直白，重点是传达清晰、符合《周易》原理。\n\n"
                                              "请根据这些要求，对以下问题进行准确回答："
        )

        prompt = [
            {
                "role": "system", "content": instruction},
            {
                "role": "user", "content": question}
        ]
        return prompt

    def ask(self, question):
        contexts = self.retriever.retrieve(question, top_k=3)

        messages = self.generate_prompt(question, contexts)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response, _ = self.model.chat(self.tokenizer, query=prompt, history=[]) #model.chat is ChatGLM implementation

        # model_inputs = self.tokenizer([prompt], return_tensors="pt").to('cuda')
        #
        # generated_ids = self.model.generate(
        #     input_ids=model_inputs.input_ids,
        #     max_new_tokens=2048#512
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        #
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

class Evaluater:
    def __init__(self, rag_model, eval_llm=None):
        self.rag_model = rag_model
        self.eval_llm = eval_llm

    def evaluate_with_reference(self, questions, references, metric="bertscore"):
        rag_answers = [self.rag_model.ask(q) for q in questions]
        if metric == "bertscore":
            P, R, F1 = bert_score(rag_answers, references,
                                  # model_type="/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea",
                                  lang="zh", rescale_with_baseline=True, verbose=True)
            return [{"question": q, "rag_answer": a, "reference": r, "bertscore": f.item()} 
                    for q, a, r, f in zip(questions, rag_answers, references, F1)]
        # ...other metrics
        else:
            raise NotImplementedError("only bertscore implemented")

    def evaluate_with_llm(self, questions, references=None, prompt_template=None):
        if self.eval_llm is None:
            raise ValueError("eval llm is None")
        rag_answers = [self.rag_model.ask(q) for q in questions]
        results = []
        for i, (q, a) in enumerate(zip(questions, rag_answers)):
            if references:
                prompt = prompt_template.format(question=q, rag_answer=a, reference=references[i])
            else:
                prompt = prompt_template.format(question=q, rag_answer=a)
            score = self.eval_llm.gpt_judge(prompt)
            results.append({"question": q, "rag_answer": a, "score": score})
        return results


def load_config(config_path: str = '/root/llm_NaiveFinetune/config/rag.yaml'):
    rag_config = yaml.safe_load(open(config_path, 'r'))
    embedding_model_path = rag_config['embedding_model']
    reranker_model_path = rag_config['reranker_model']
    return embedding_model_path, reranker_model_path

def main():
    document_dir = '/root/llm_NaiveFinetune/data/raw'
    model_name_or_path = "/root/autodl-tmp/huggingface/hub/models--THUDM--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac"
    peft_model_path = "/resources/epoch10-20250709_155029/checkpoint-200"
    dataset_path = '/root/llm_NaiveFinetune/data/zhouyi_dataset_20240118_163659.csv'
    from prompt_templates import llm_prompt_template, llm_prompt_template_no_ref

    embedding_model_path, reranker_model_path = load_config()

    rag = LLM_w_RAG(document_dir, model_name_or_path, peft_model_path, embedding_model_path, reranker_model_path)
    ai_judge = Judges()

    with open("/root/llm_NaiveFinetune/data/questions_references4rag.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        questions = data["questions"]
        references = data["references"]

    evaluator = Evaluater(rag_model=rag, eval_llm=ai_judge)
    bertscore_results = evaluator.evaluate_with_reference(questions, references)

    llm_results = evaluator.evaluate_with_llm(
        questions,
        references,
        prompt_template=llm_prompt_template
    )

    evaluation_data = {
        "evaluation_info": {
            "model_name": model_name_or_path,
            "peft_model_path": peft_model_path,
            "embedding_model_path": embedding_model_path,
            "dataset_path": dataset_path,
            "sample_size": len(questions),
            "evaluation_timestamp": pd.Timestamp.now().isoformat()
        },
        "bertscore_evaluation": bertscore_results,
        "llm_evaluation": llm_results,
        "questions": questions,
        "references": references
    }

    output_file = f"evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到: {output_file}")

    print("\n=== 评估摘要 ===")
    print(f"BERTScore平均分: {sum(r['bertscore'] for r in bertscore_results) / len(bertscore_results):.4f}")

    llm_scores = []
    for result in llm_results:
        score_text = result['score']
        try:
            score_line = [line for line in score_text.split('\n') if line.startswith('分数:')][0]
            score = int(score_line.split(':')[1].strip())
            llm_scores.append(score)
        except:
            llm_scores.append(None)

    valid_scores = [s for s in llm_scores if s is not None]
    if valid_scores:
        print(f"LLM主观评分平均分: {sum(valid_scores) / len(valid_scores):.2f}")

    print(f"详细结果请查看: {output_file}")

if __name__ == '__main__':
    main()