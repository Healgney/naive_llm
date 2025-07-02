from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sklearn.metrics import f1_score
from bert_score import score as bert_score

#TODO: 多模态Retrieve（图片）

class DocProcessor:
    def __init__(self, data_path:str='data'):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 16}
        )
        self.data_path = data_path
        self.split_params = {
            "code":   {"chunk_size": 256, "chunk_overlap": 64},
            "table":  {"chunk_size": 384, "chunk_overlap": 96},
            "normal": {"chunk_size": 128, "chunk_overlap": 24}
        }

    def _detect_content_type(self, text):
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:
            return "table"
        return "normal"
    
    def _get_splitter(self, content_type):
            params = self.split_params.get(
                content_type,
                self.split_params["normal"] #default
            )
            return RecursiveCharacterTextSplitter(**params)

    def process_documents(self):
        loaders = [
            DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        chunker = SemanticChunker(
            embeddings=self.embed_model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)

        final_chunks = []
        # for chunk in base_chunks:
        #     content_type = self._detect_content_type(chunk.page_content)
        #     if content_type == "code":
        #         splitter = RecursiveCharacterTextSplitter(
        #             chunk_size=256, chunk_overlap=64)
        #     elif content_type == "table":
        #         splitter = RecursiveCharacterTextSplitter(
        #             chunk_size=384, chunk_overlap=96)
        #     else:
        #         splitter = RecursiveCharacterTextSplitter(
        #             chunk_size=128, chunk_overlap=24)
        #     final_chunks.extend(splitter.split_documents([chunk]))

        # for i, chunk in enumerate(final_chunks):
        #     chunk.metadata.update({
        #         "chunk_id": f"chunk_{i}",
        #         "content_type": self._detect_content_type(chunk.page_content)
        #     })

        for i, chunk in enumerate(base_chunks):
            content_type = self._detect_content_type(chunk.page_content)
            splitter = self._get_splitter(content_type)
            sub_chunks = splitter.split_documents([chunk])
            for sub_chunk in sub_chunks:
                sub_chunk.metadata.update({
                    "chunk_id": f"chunk_{len(final_chunks)}",
                    "content_type": content_type
                })
                final_chunks.append(sub_chunk)


        return final_chunks


class Retriever:
    def __init__(self, chunks):
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-zh-v1.5" #bge-small-zh
            ),
            persist_directory="./vector_db"
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
            "BAAI/bge-reranker-large",
            device="cuda" if torch.has_cuda else "cpu"
        )

    def retrieve(self, query, top_k=3):
        docs = self.ensemble_retriever.get_relevant_documents(query)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:top_k]]


class EnhancedRAG:
    def __init__(self, model_name_or_path, cache_dir=None):
        self.model_name_or_path = model_name_or_path

        processor = DocProcessor()
        chunks = processor.process_documents()

        self.retriever = Retriever(chunks)
        if cache_dir is not None:
            self.cache_dir = cache_dir
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to('cuda')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            ).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.model.eval()

    def generate_prompt(self, question, contexts):
        context_str = "\n\n".join([
            f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
            for doc in contexts
        ])
        instruction = (
                "你是一名专业法律助手，请严格根据以下刑法条文依据进行分析,不要违背以下刑法依据：\n\n"
                f"{'#' * 7}\n".join(context_str) + "\n\n"
                                                   "回答要求：\n"
                                                   "1. 准确引用刑法法律条文\n"
                                                   "2. 分析案件与法条的对应关系\n"
                                                   "3. 结论明确，使用法律术语\n"
                                                   "请按照上述要求回答以下法律问题："
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

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class Evaluater:
    def __init__(self, rag_model, eval_llm=None):
        """
        rag_model: EnhancedRAG 实例
        eval_llm: 可选，评测用的大模型（如ChatGPT API封装），用于主观打分
        """
        self.rag_model = rag_model
        self.eval_llm = eval_llm

    def evaluate_with_reference(self, questions, references, metric="bertscore"):
        """
        questions: 问题列表
        references: 参考答案列表
        metric: "bertscore" 或 "rouge" 等
        返回每个问题的分数
        """
        rag_answers = [self.rag_model.ask(q) for q in questions]
        if metric == "bertscore":
            P, R, F1 = bert_score(rag_answers, references, lang="zh", rescale_with_baseline=True)
            return [{"question": q, "rag_answer": a, "reference": r, "bertscore": f.item()} 
                    for q, a, r, f in zip(questions, rag_answers, references, F1)]
        # 可扩展更多指标
        else:
            raise NotImplementedError("只实现了bertscore")

    def evaluate_with_llm(self, questions, references=None, prompt_template=None):
        """
        用大模型主观评测RAG答案质量
        prompt_template: 可选，str，插入{question}、{rag_answer}、{reference}
        """
        if self.eval_llm is None:
            raise ValueError("未提供评测用大模型")
        rag_answers = [self.rag_model.ask(q) for q in questions]
        results = []
        for i, (q, a) in enumerate(zip(questions, rag_answers)):
            if references:
                prompt = prompt_template.format(question=q, rag_answer=a, reference=references[i])
            else:
                prompt = prompt_template.format(question=q, rag_answer=a)
            score = self.eval_llm(prompt)  # 假设eval_llm返回分数或评价
            results.append({"question": q, "rag_answer": a, "score": score})
        return results


def main():
    model_name_or_path = "../sft_merged_model"
    # cache_dir="../Qwen_model_file"
    rag = EnhancedRAG(model_name_or_path)
    question1 = "崔某作为个体工商户，在税务人员征税过程中采取暴力抗拒行为，并故意将热油泼向围观群众，导致一名群众重伤。请问上述行为是否构成故意伤害罪？请给出详细的推理过程之后再给出答案。"
    question2 = "小明冒用了别人的信用卡进行消费，数额较大，是否构成信用卡诈骗罪？"
    question3 = "某公司将大量液态废物走私进境，企图从中获取不当利益。海关查获后，该公司及其负责主管人员被起诉。对于该公司及其负责主管人员，按照什么规定处罚走私液态废物的行为？"
    question4 = "小明是某银行的员工，他利用职务上的便利，收受客户回扣，将钱财占为己有。小明的行为构成什么罪？应该如何处罚？"
    question5 = "甲公司为了获取国有企业A的业务，在接触中向A公司领导的亲戚赠送大量礼物，后甲公司成功获得A公司的业务。根据相关法律法规，甲公司是否已经违法？"
    prompt = [question1, question2, question3, question4, question5]
    for i in range(len(prompt)):
        answer = rag.ask(prompt[i])
        print(f"prompt{i + 1}：{prompt[i]}")
        print(f"response：{answer}")

    # rag = EnhancedRAG(...)
    questions = [...]
    references = [...]
    evaluator = Evaluater(rag_model=rag)
    bertscore_results = evaluator.evaluate_with_reference(questions, references)
    print(bertscore_results)

    # 如果有 LLM 评测接口
    # def gpt_judge(prompt): ...
    # evaluator = Evaluater(rag_model=rag, eval_llm=gpt_judge)
    # results = evaluator.evaluate_with_llm(questions, references, prompt_template="请为以下问答打分...")


if __name__ == '__main__':
    main()