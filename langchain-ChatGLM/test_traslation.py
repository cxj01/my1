from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

pipeline = pipeline("translation", model=AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en"), tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"))
chinese = """
六岁时，我家在荷兰的莱斯韦克，房子的前面有一片荒地，
我称其为“那地方”，一个神秘的所在，那里深深的草木如今只到我的腰际，
当年却像是一片丛林，即便现在我还记得：“那地方”危机四伏，
洒满了我的恐惧和幻想。
"""
# chinese = 'When I was six years old, my family was in Leiswick in the Netherlands, and there was a wasteland in front of the house, and I called it "the place," a mysterious place where the deep grass was now on my waist, and it was like a jungle, even now that I remember: "The place," full of my fears and illusions.'
result = pipeline(chinese)
print(result[0]['translation_text'])