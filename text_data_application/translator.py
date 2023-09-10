from googletrans import Translator
import pandas as pd


translator = Translator()

def translate(sentences: pd.DataFrame):
    try:
        # 번역
        translated_sentences = []
        for sentence in sentences['title-ko']:
            translated_sentences.append(translator.translate(str(sentence), 'en', 'ko').text)

        # 데이터 프레임에 열 추가
        sentences["title-en"] = translated_sentences
    except TypeError as e:
        print("----------------------------------")
        print("Exception Occured:", e)
        print("----------------------------------")
    return sentences