from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter
import re
from typing import List

class NewSpacyTextSplitter(SpacyTextSplitter):
    def __init__(self, pdf, **kwargs):
        self.pdf = pdf
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        splits = (s.text for s in self._tokenizer(text).sents)
        return self._merge_splits(splits, self._separator)



# if __name__ == "__main__":
#     import sys
#     filepath = "/data/cxj_workplace/data/test_en_pdf/953.pdf"
#     # filepath = "/data/cxj_workplace/data/俄乌/2月24日之后斯拉夫欧亚的危机_重访晚期苏联_松里公孝.pdf"
#     loader = PyPDFLoader(filepath)
#     # text_split = AllTextSplitter(pdf=True, chunk_size=250, chunk_overlap=100)
#     # text_split = RecursiveCharacterTextSplitter(
#     #     chunk_size=450,
#     #     chunk_overlap=100,
#     # )
#     # text_split = SpacyTextSplitter(
#     #     pipeline="zh_core_web_md",
#     #     # pipeline="en_core_web_sm",
#     #     chunk_size=250,
#     #     chunk_overlap=50,
#     # )
#
#     text_split = NewSpacyTextSplitter(
#         pdf=True,
#         pipeline="zh_core_web_md",
#         separator=" ",
#         # pipeline="en_core_web_sm",
#         chunk_size=450,
#         chunk_overlap=120,
#     )
#     docs = loader.load_and_split(text_split)
#
#     # docs = loader.load()
#     for doc in docs:
#         print(doc)