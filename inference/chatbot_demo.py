
import gradio as gr
from peft import PeftModel, PeftConfig
import mdtex2html

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import json

# 1️⃣ 加载检索模型（embedding）
embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 2️⃣ 加载向量索引
index = faiss.read_index("data/zhouyi_index.faiss")
with open("data/zhouyi_meta.json", "r", encoding="utf-8") as f:
    meta_data = json.load(f)

model_path = "/home/healgney/Documents/huggingface/hub/models--resources--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac"
peft_model_path = "/home/healgney/Documents/LLM_NaiveFinetune_demo/models/chatglm3-6b-epoch3-20250611_230752/checkpoint-50"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()

"""Override Chatbot.postprocess"""

# 4️⃣ 构造带检索信息的 prompt
def build_prompt(contexts, user_question):
    context_str = "\n".join([f"【知识】{c['text']}" for c in contexts])
    prompt = f"{context_str}\n\n【问题】{user_question}\n【回答】"
    return prompt

# 5️⃣ 检索模块
def retrieve_docs(query, top_k=3):
    query_emb = embed_model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [meta_data[i] for i in I[0]]

# 6️⃣ 主函数：RAG 推理
def rag_ask(question):
    related_docs = retrieve_docs(question)
    prompt = build_prompt(related_docs, question)

    response, _ = model.chat(tokenizer, prompt, history=[])
    return response

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot(height=1000)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)

# 7️⃣ 测试
if __name__ == "__main__":
    query = "请解释蒙卦中“童蒙求我”的意思"
    answer = rag_ask(query)
    print(f"\n🧠 问题：{query}\n📘 回答：{answer}")