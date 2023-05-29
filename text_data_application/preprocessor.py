from googletrans import Translator
import pandas as pd


translator = Translator()

def preprocesser(sentences: pd.DataFrame):
    try:
        translated_sentences = []
        for sentence in sentences['title-ko']:
            translated_sentences.append(translator.translate(str(sentence), 'en', 'ko').text)

        sentences["title-en"] = translated_sentences
    except TypeError as e:
        print("Exception Occured:", e)
    return sentences