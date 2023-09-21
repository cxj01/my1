import time

import gradio as gr
import shutil

import requests
from elasticsearch import Elasticsearch

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.base import AnswerResult
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os
import json

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
    es_index_config = json.load(f)

# index_to_name = es_index_config["index_to_name"]
#
# name_to_index = es_index_config["name_to_index"]

def get_es_index_to_name():
    with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
        es_index_config = json.load(f)
    return es_index_config["index_to_name"]

def get_es_name_to_index():
    with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
        es_index_config = json.load(f)
    return es_index_config["name_to_index"]

def get_vs_list():
    lst_default = ["新建知识库"]
    with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
        es_index_config = json.load(f)
    index_to_name = es_index_config["index_to_name"]
    return lst_default + list(index_to_name.values())

def create_es_index(index):
    mappings = {
        "properties": {
          "metadata": {
            "properties": {
              "category": {
                "type": "keyword"
              },
              "file_directory": {
                "type": "keyword"
              },
              "filename": {
                "type": "keyword"
              },
              "filetype": {
                "type": "keyword"
              },
              "page_number": {
                "type": "long"
              },
              "source": {
                "type": "keyword"
              }
            }
          },
          "text": {
            "type": "text",
            "analyzer": "ik_smart"
          },
          "vector": {
            "type": "dense_vector",
            "dims": 1024,
            "index": False
          }
    }
    }
    settings = {
    "analysis": {
      "analyzer": {
        "ik_smart": {
          "type": "custom",
          "tokenizer": "ik_smart"
        },
        "ik_max_word": {
          "type": "custom",
          "tokenizer": "ik_max_word"
        }
      }
    }
  }
    es_client = Elasticsearch(hosts=[ELASTICSEARCH_URL], verify_certs=False)
    es_client.indices.create(index=index, mappings=mappings, settings=settings)

def delete_es_index(index):
    es_client = Elasticsearch(hosts=[ELASTICSEARCH_URL], verify_certs=False)
    es_client.indices.delete(index=index)


def delete_es_index_files(index, files):
    es_client = Elasticsearch(hosts=[ELASTICSEARCH_URL], verify_certs=False)
    query = {
      "query": {
        "bool": {
          "must": [
            {
              "terms": {
                "metadata.filename": files
              }
            }
          ]
        }
      }
    }
    es_client.delete_by_query(index=index, body=query)
    es_client.indices.refresh(index=index)


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def process_outline(vs_path, files_to_delete):
    content_path = os.path.join(ES_ROOT_PATH, vs_path, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    print(docs_path)
    # ss = "/data/cxj_workplace/jupyterlab/somethings/requestT/1.docx"
    if os.path.exists(docs_path[0]):
        r = requests.post('http://10.1.226.1:9090/outlineJson', files={"file": open(docs_path[0], 'rb')})
        data = r.json()
        result = ""
        for i in range(len(data)):
            item = data[i]
            result += "<p>" + "#" * (item["level"] + 1) + item["headline"] + "</p>"
        return result
    else:
        return "<p>" + "ERROR" + "</p>"


def get_answer(query, vs_path, history, mode, files_to_delete, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if type(vs_path) == type(gr.State()):
        vs_path = vs_path.value
    # vs_path = name_to_index[vs_path]
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "专用知识问答" and vs_path is not None and vs_path in get_vs_list():
        vs_path = get_es_name_to_index()[vs_path]
        if len(files_to_delete) == 0:
            for resp, history in local_doc_qa.get_knowledge_based_answer_by_es(
                    query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
                source = "\n\n"
                source += "".join(
                    [f"""<details> <summary>出处 [{i + 1}] {"_".join(os.path.split(doc[0].metadata["source"])[-1].split("_")[0: 1 if(len(os.path.split(doc[0].metadata["source"])[-1].split("_"))) == 1 else len(os.path.split(doc[0].metadata["source"])[-1].split("_")) - 1]) + str(os.path.splitext(os.path.split(doc[0].metadata["source"])[-1])[1] if(len(os.path.split(doc[0].metadata["source"])[-1].split("_"))) > 1 else "")}</summary>\n"""
                     f"""{"".join(doc[0].page_content.split(SPECIAL_FLAG)[-1])}\n"""
                     f"""</details>"""
                     for i, doc in
                     enumerate(resp["source_documents"])])
                history[-1][-1] += source
                yield history, ""
        else:
            if "大纲" in query:
                # s = "qewewoeweorfoieioe"
                s = process_outline(vs_path, files_to_delete)
                history += [[]]
                pre = ""
                for ans in iter(s):
                    t_resp = ans
                    pre += ans
                    history[-1] = [query, t_resp]
                    time.sleep(0.01)
                    history[-1][-1] = pre
                    yield history, ""
            else:

                for resp, history in local_doc_qa.get_knowledge_based_answer_by_es_files(
                        query=query, vs_path=vs_path, chat_history=history, file_list=files_to_delete, streaming=streaming):
                    source = "\n\n"
                    source += "".join(
                        [f"""<details> <summary>出处 [{i + 1}] {"_".join(os.path.split(doc[0].metadata["source"])[-1].split("_")[0: 1 if(len(os.path.split(doc[0].metadata["source"])[-1].split("_"))) == 1 else len(os.path.split(doc[0].metadata["source"])[-1].split("_")) - 1]) + str(os.path.splitext(os.path.split(doc[0].metadata["source"])[-1])[1] if(len(os.path.split(doc[0].metadata["source"])[-1].split("_"))) > 1 else "")}</summary>\n"""
                         f"""{"".join(doc[0].page_content.split(SPECIAL_FLAG)[-1])}\n"""
                         f"""</details>"""
                         for i, doc in
                         enumerate(resp["source_documents"])])
                    history[-1][-1] += source
                    yield history, ""

    elif mode == "知识库测试":
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def init_model():
    args = parser.parse_args()

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(ES_ROOT_PATH, vs_id, "vector_store")
    vs_id = get_es_name_to_index()[vs_id]
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(ES_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(ES_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.es_init_knowledge_vector_store(filelist, vs_id, sentence_size)
            vs_path = get_es_index_to_name()[vs_path]
        else:
            vs_path, loaded_files = local_doc_qa.es_one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
            vs_path = get_es_index_to_name()[vs_path]
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.es_list_file_from_vector_store(get_es_name_to_index()[vs_path]) if vs_path else [])


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history,\
                gr.update(choices=[]), gr.update(visible=False)
    else:
        # vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if vs_id in get_vs_list():
        # if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_id, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.es_list_file_from_vector_store(get_es_name_to_index()[vs_id]), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_id, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")


def change_mode(mode, history):
    if mode == "专用知识问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, vs_zh_name, chatbot):
    if vs_name is None or vs_name.strip() == "" or vs_zh_name is None or vs_zh_name.strip() == "":
        vs_status = "知识库名称不能为空，请重新填写知识库名称"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    elif vs_zh_name in get_vs_list():
        vs_status = vs_zh_name + "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    elif vs_name in list(get_es_name_to_index().values()):
        vs_status = vs_name + "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(ES_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(ES_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(ES_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(ES_ROOT_PATH, vs_name, "vector_store"))
        create_es_index(vs_name)
        with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
            temp_data = json.load(f)
        temp_data["index_to_name"][vs_name] = vs_zh_name
        temp_data["name_to_index"][vs_zh_name] = vs_name
        with open(ES_INDEX_CONFIG_PATH, "w", encoding="utf-8")as f:
            f.write(json.dumps(temp_data, ensure_ascii=False, indent=4))
        vs_status = f"""已新增知识库"{vs_zh_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_zh_name), gr.update(
            visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(ES_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(ES_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(ES_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())

def cxj_refresh_vs_list():
    return gr.update(choices=get_vs_list())

def delete_file(vs_id, files_to_delete, chatbot):
    vs_id = get_es_name_to_index()[vs_id]
    vs_path = os.path.join(ES_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(ES_ROOT_PATH, vs_id, "content")
    try:
        docs_path = [os.path.join(content_path, file) for file in files_to_delete]
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
        delete_es_index_files(vs_id, files_to_delete)
        vs_status = "文件删除成功。"
    except Exception as e:
        print(e.__str__())
        vs_status = "文件删除失败。"

    logger.info(",".join(files_to_delete)+vs_status)
    chatbot = chatbot + [[None, vs_status]]
    choices = local_doc_qa.es_list_file_from_vector_store(vs_id)
    return gr.update(choices=choices, value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        with open(ES_INDEX_CONFIG_PATH, "r", encoding="utf-8")as f:
            temp_data = json.load(f)
        temp_index = temp_data["name_to_index"][vs_id]
        del(temp_data["name_to_index"][vs_id])
        del(temp_data["index_to_name"][temp_index])
        with open(ES_INDEX_CONFIG_PATH, "w", encoding="utf-8")as f:
            f.write(json.dumps(temp_data, ensure_ascii=False, indent=4))
        delete_es_index(temp_index)
        shutil.rmtree(os.path.join(ES_ROOT_PATH, temp_index))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉HF-GPT Web UI🎉
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""欢迎使用 HF-GPT Web UI！

请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

知识库问答模式，选择知识库名称后，即可开始问答，当前知识库{default_vs}，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。

知识库暂不支持文件删除，该功能将在后续版本中推出。
"""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status, model_status = gr.State(
        get_vs_list()[0] if len(get_vs_list()) > 1 else ""), gr.State(""), gr.State(
        model_status)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["通用知识问答", "专用知识问答"],
                                label="请选择使用模式",
                                value="专用知识问答", )
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项", visible=True)
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                    vs_zh_name = gr.Textbox(label="请输入新建知识库的中文名称",
                                             lines=1,
                                             interactive=True,
                                             visible=True)
                    vs_name = gr.Textbox(label="请输入新建知识库的英文名称，并且需要以fg_为开头",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    vs_delete = gr.Button("删除本知识库", visible=False)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("选择文件"):
                            files_to_delete = gr.CheckboxGroup(choices=[],
                                                             label="请从知识库已有文件中选择特定文件进行问答",
                                                             interactive=True)
                            delete_file_button = gr.Button("从知识库中删除选中文件", visible=True)
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, vs_zh_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_zh_name, vs_add, file2vs, chatbot, vs_delete])
                    vs_delete.click(fn=delete_vs,
                                    inputs=[select_vs, chatbot],
                                    outputs=[select_vs, vs_name, vs_zh_name, vs_add, file2vs, chatbot, vs_delete])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_zh_name, vs_add, file2vs, vs_path, chatbot, files_to_delete, vs_delete])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, files_to_delete], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, files_to_delete], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, files_to_delete],
                                 [chatbot, query])
                    delete_file_button.click(delete_file,
                                             show_progress=True,
                                             inputs=[select_vs, files_to_delete, chatbot],
                                             outputs=[files_to_delete, chatbot])

    demo.load(
        fn=cxj_refresh_vs_list,
        inputs=None,
        outputs=[select_vs],
        queue=True,
        show_progress=False,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=12305,
         show_api=False,
         share=False,
         inbrowser=False))
