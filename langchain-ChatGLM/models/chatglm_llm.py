from abc import ABC

from bs4 import BeautifulSoup
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)



def process_html_p(item):
    if '<details> <summary>' in item:
        soup = BeautifulSoup(item, 'html.parser')
        # 获取所有 <p> 标签内的字符串内容
        p_texts = [tag.get_text() for tag in soup.find_all('p')]
        # 输出所有 <p> 标签内的字符串内容
        result = ""
        for text in p_texts:
            result += text
        return result
    else:
        return item

def process_history(query, history):
    result = []
    if '。' != query[-1]:
        return result
    else:
        temp = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i][0]:
                if '。' not in history[i][0]:
                    temp = i
                    break
        if temp == -1:
            return result
        else:
            for i in range(temp, len(history)):
                result.append([history[i][0], process_html_p(history[i][1])])
        return result

class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):
        nh = process_history(prompt, history)
        if len(nh):
            prompt = prompt.split("问题是：")[-1]
            # PROMPT_TEMPLATE = """已知信息：
            # {context}
            #
            # 根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""
            # prompt = PROMPT_TEMPLATE.replace("{question}", question).replace("{context}", context)
        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=nh,
                    max_length=self.max_token,
                    temperature=self.temperature,
                    repetition_penalty=1.2
            )):
                # self.checkPoint.clear_torch_cache()
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                yield answer_result
        else:
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=nh,
                max_length=self.max_token,
                temperature=self.temperature
            )
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            yield answer_result


