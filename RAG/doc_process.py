from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


class DocProcessor:
    def __init__(self, data_path:str = 'data', embd_model:str=''):
        self.embed_model = HuggingFaceEmbeddings(
            model_name=embd_model,#"BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 16},
            # cache_folder='/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-zh-v1.5'
        )
        self.data_path = data_path
        self.split_params = {
            # "code": {"chunk_size": 256, "chunk_overlap": 64},
            # "table": {"chunk_size": 384, "chunk_overlap": 96},
            # "normal": {"chunk_size": 128, "chunk_overlap": 24},
            "normal": {
                    'chunk_size':400,                                   # 每块约250~300个汉字，适合保持语义完整性
                    'chunk_overlap':50,                                 # 保证跨段时不会断裂上下文
                    'separators':["\n\n", "\n", "。", "；", "，", " "]  # 从结构性断点逐级回退
                }

        }

    def _detect_content_type(self, text):
        # if re.search(r'def |import |print\(|代码示例', text):
        #     return "code"
        # elif re.search(r'\|.+\|', text) and '%' in text:
        #     return "table"
        return "normal"

    def _get_splitter(self, content_type):
        params = self.split_params.get(
            content_type,
            self.split_params["normal"]  # default
        )
        return RecursiveCharacterTextSplitter(**params)

    def process_documents(self):
        loaders = [
            # DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
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