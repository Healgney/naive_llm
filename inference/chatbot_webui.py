import gradio as gr

from langchain_community.llms import ChatGLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel
import torch

CHATGLM_URL = "http://127.0.0.1:8000"

# def init_chatbot():
#     llm = ChatGLM(
#         endpoint_url=CHATGLM_URL,
#         max_token=80000,
#         history=[],
#         top_p=0.9,
#         model_kwargs={"sample_model_args": False},
#     )
#     global CHATGLM_CHATBOT
#     CHATGLM_CHATBOT = ConversationChain(llm=llm,
#                                         verbose=True,
#                                         memory=ConversationBufferMemory())
#     return CHATGLM_CHATBOT
#
# def chatglm_chat(message, history):
#     ai_message = CHATGLM_CHATBOT.predict(input = message)
#     return ai_message

tokenizer = None
base_model = None

def init_chatbot():
    global tokenizer, base_model
    model_path = "/home/healgney/Documents/huggingface/hub/models--resources--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac"
    peft_model_path = "/home/healgney/Documents/LLM_NaiveFinetune_demo/models/chatglm3-6b-epoch3-20250611_230752/checkpoint-50"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    base_model = PeftModel.from_pretrained(base_model, peft_model_path)
    base_model.eval()
    print("模型加载完成")


# def chatglm_chat(message, history):
#     global tokenizer, base_model
#     if history is None:
#         history = []
#     response, _ = base_model.chat(tokenizer, message, history)
#     history.append((message, response))
#     return history

def chatglm_chat(message, history):
    global tokenizer, base_model

    if history is None:
        history = []

    # 给模型用的历史格式
    model_history = history

    # 获取回复（注意不要覆盖 history）
    response, _ = base_model.chat(tokenizer, message, model_history)

    # 正确追加新的对话元组
    history.append((message, response))

    return history

def launch_gradio():
    demo = gr.ChatInterface(
        type='messages',
        fn=chatglm_chat,
        title="ChatBot (Powered by ChatGLM)",
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")
    # demo.launch(
    #     # server_name="0.0.0.0", server_port=7860
    # )


if __name__ == "__main__":
    init_chatbot()
    launch_gradio()
