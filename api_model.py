# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:29:24 2019

@author: Nayan
"""
class model():

    def __init__(self):
        from models.TextRankModels import TextRank4Keyword, TextRank4Keyphrase
        import spacy
        self.stop_words = []
        self.nlp = spacy.load("en_core_web_sm")
        self.max_no_of_kw_kp = 20
        self.lower = False
        self.window_size = 5
        self.trkw = TextRank4Keyword()
        self.trkp = TextRank4Keyphrase()
    
    def get_kw_kp_large_text(self,text):
        """
        Function to get keywords and phrases : using textrank (spacy and nltk)
        Targetted POS tags, and maximum number of keywords extracted can be changed here.
        INPUT:
            text : text to extract keywords and phrases
            max_no_of_kw_kp : Maximum no of keywords and keyphrases to extract from a text
            lower : True / False
            stop_words : list of stopwords
            curse_words : list of curse_words
        OUTPUT:
            list of keywords and keyphrases. 
        """
        keywords = []
        #get key-word list
        self.trkw.analyze(text)
        key = self.trkw.get_keywords(number = self.max_no_of_kw_kp)
        [keywords.append(i) for i in list(key.keyword)]
        self.trkp.stop_words =self.stop_words
        self.trkp.n_keywords_percent=1-(self.max_no_of_kw_kp/100)
        #get key-phrase list
        keyph = self.trkp.score_keyphrases_by_textrank(text)
    
        for key in keyph:
            keywords.append(list(key)[0])
               
        result_list = list(set(keywords))
    
        return result_list
    
    def get_kw_kp_small_text(self,text):
        """
        Function to get keywords and phrases for small texts (contains less than 20 words)
        INPUT:
            text : text to extract keywords and phrases
            lower : True / False
        OUTPUT:
            list of keywords and keyphrases. 
        """
        
        #import json
        import nltk
        from nltk.tokenize import word_tokenize
        pos_list =  {
                'spacy' : ['NOUN', 'PROPN','VERB','ADJ','ADV','X'],
                'nltk' : ['VB','RB','RBR','RBS','JJ','JJR','JJS','NN','NNP','NNS','NNPS']
                }
        #doc = nlp(text)
        
        word_list = []
        #text = " ".join(word_list)
        phrase_list = []
        for t in text.split("."):#get all word in a sentence
            doc = self.nlp(t)
            
            #get word list
            [word_list.append(token.text) for token in doc if token.pos_ in pos_list.get('spacy') and token.is_stop is False]
            
            #get phrase list
            tokens = word_tokenize(" ".join([token.text for token in doc if token.pos_ in pos_list.get('spacy') and token.is_stop is False]))
            bigrm = nltk.bigrams(tokens)
            [phrase_list.append(phrase) for phrase in [*map(' '.join, bigrm)]]
        
        return word_list + phrase_list
    
    def predict(self,input_data):
        
        text = input_data
        # 2. process input with simple tokenization and no punctuation
        if len(str(text).split(" ")) > 20:#get key-word, key-phrases from the text that contain more than 20 words
            word_list = self.get_kw_kp_large_text(text)
        
        else:
            word_list = self.get_kw_kp_small_text(text)
    
        # 4. process the output
        output_data = {"input": text, "output": word_list}
        # 5. return the output for the api
        return output_data
