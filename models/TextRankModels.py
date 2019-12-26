# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:33:37 2019

@author: Nayan
"""

class TextRank4Keyword():
    
    """
    Extract keywords from text
    
    Functions :
        set_stopwords
        sentence_segment
        get_vocab
        get_token_pairs
        symmetrize
        get_keywords
        analyse
    """
    
    import numpy as np
    import spacy
    
    def __init__(self):
        """
        Parameters:
            d : damping coefficient, usually is .85
            min_diff : convergence threshold
            steps : iteration steps
            node_weight : to save keywords and its weight
        """
        self.d = 0.85
        self.min_diff = 1e-5 
        self.steps = 10
        self.node_weight = None
        self.non_stopwords = []
        self.candidate_pos = ['NOUN', 'PROPN','VERB','ADJ','ADV','X']
        self.window_size = 5
        self.lower = False
        self.stopwords = []
    
    def set_stopwords(self, stopwords):
        """
        Include stop words
        
        INPUT:
            stopwords : A list of stopwords
            non_stopwords :  A list of words to exclude from stopword list
        OUTPUT:
            nlp : a spacy small model instanace with stopwords included  
        """
        from spacy.lang.en.stop_words import STOP_WORDS
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        for word in STOP_WORDS.union(set(stopwords)): #Getting those stop words that are not included already
            nlp.Defaults.stop_words.add(word.lower())
        for word in self.non_stopwords:
            nlp.vocab[word].is_stop = False
        return nlp
    
    def sentence_segment(self, doc, lower):
        """
        Store those words only in cadidate_pos
        
        INPUT:
            doc : text document 
            lower : True / False
        OUTPUT:
           sentences: a list of sentences 
        """
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in self.candidate_pos and token.is_stop is False:  # select those words with given pos and which are not stop words
                    if lower is True: #To select only lower words
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences #return a list of lists 
        
    def get_vocab(self, sentences): #tokenizes each word in sentences
        """
        Get all tokens
        
        INPUT:
            sentences : list of sentences
        OUTPUT:
            vocab: a orderded dictionary of vacab prepared from sentences.
        """
        from collections import OrderedDict
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, sentences):
        """
        Build token_pairs from windows in sentences
        
        INPUT:
            window_size : no of words to be taken together
            sentences : list of sentences
        OUTPUT:
            token_pair: words changed to tokens returned as a pair
        """
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        """symmetrize a matrix
        
        INPUT:
            a: a matirx
        OUTPUT:
            a symmetrized matrix
        """
        import numpy as np
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix
        
        INPUT:
            vocab: vacab prepared from the text
            token_pair: token pair prepared from the sentences.
        OUTPUT:
            g_norm: a matrix
        """
        import numpy as np
        # Build matrix
    
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number):
        """Print top number keywords
        
        INPUT:
            number: Maximum number of keywords to be fetched
        OUTPUT:
            a dataframe containing keyword and its weight
        """
        import pandas as pd
        from collections import OrderedDict
        
        k, v = [], [] 
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            #print(key + ' - ' + str(value))
            k.append(key)
            v.append(value)
            if i > number:
                break
        return pd.DataFrame({'keyword':k,'value':v})
        
        
    def analyze(self, text):
        """Main function to analyze text
        
        INPUT:
            text: the text to be analyzed
            window_size: no of words to be taken together
            lower: True / False
            stopwords: stopwords to be removed
        OUTPUT:
            None
        """
        import spacy
        import numpy as np
        nlp = spacy.load("en_core_web_sm")
        # Set stop words
        nlp = self.set_stopwords(self.stopwords)
        
        # Pare text by spaCy
        doc =nlp(text)
        # Filter sentences
        sentences = self.sentence_segment(doc, lower = self.lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        #print(vocab)
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

###############################################################################

"""Textrank for keyphrase"""

class TextRank4Keyphrase():
    
    def __init__(self):
        self.stop_words = [None] 
        self.candidate_pos = set(['VB','RB','RBR','RBS','JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
        self.n_keywords_percent = 0.9
        self.non_stopwords = []
    
    def extract_candidate_words(self,text):
        from nltk.stem import PorterStemmer
        import itertools, nltk, string
        #setup_environment()
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        ps = PorterStemmer()
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(self.stop_words)
        Stop_words = set([i for i in stopwords if i not in self.non_stopwords])
    
        # tokenize and POS-tag words
        tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents([nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)],lang='eng'))
        # filter on certain POS tags and lowercase all words
        candidates = [ps.stem(word).lower() for word, tag in tagged_words
                  if tag in self.candidate_pos and word.lower() not in Stop_words
                  and not all(char in punct for char in word)]

        return candidates
    
    def score_keyphrases_by_textrank(self,text):
        from itertools import takewhile, tee
        import networkx, nltk
        
        
        # tokenize for all words, and extract *candidate* words
        words = [word.lower()
                 for sent in nltk.sent_tokenize(text)
                 for word in nltk.word_tokenize(sent)]
        candidates = self.extract_candidate_words(text)
        # build graph, each node is a unique candidate
        graph = networkx.Graph()
        graph.add_nodes_from(set(candidates))
        # iterate over word-pairs, add unweighted edges into graph
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
        for w1, w2 in pairwise(candidates):
            if w2:
                graph.add_edge(*sorted([w1, w2]))
        # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
        ranks = networkx.pagerank(graph)
        n_keywords_percent = self.n_keywords_percent
        if 0 < self.n_keywords_percent < 1:
            n_keywords_percent = int(round(len(candidates) * n_keywords_percent))
        word_ranks = {word_rank[0]: word_rank[1]
                      for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords_percent]}
        keywords = set(word_ranks.keys())
        # merge keywords into keyphrases
        keyphrases = {}
        j = 0
        for i, word in enumerate(words):
            if i < j:
                continue
            if word in keywords:
                kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
                avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
                keyphrases[' '.join(kp_words)] = avg_pagerank
                # counter as hackish way to ensure merged keyphrases are non-overlapping
                j = i + len(kp_words)
        
        return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)

###############################################################################