from googletrans import Translator
import pandas as pd


translator = Translator()

def preprocesser(sentences: pd.DataFrame):
    translated_sentences = []
    for sentence in sentences['title']:
        translated_sentences.append(translator.translate(str(sentence), 'en', 'ko').text)

    sentences.rename(columns={"title" : "title-ko"}, inplace=True)
    sentences["title-en"] = translated_sentences
    
    return sentences