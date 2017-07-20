# author : paul ogier
# coding=utf-8
import pandas as pd
import numpy as np
import neatcleanstring as ncs


class word_bagging_table:
    def __init__(self, load_existing=pd.DataFrame()):
        if load_existing.shape[1]==0:
            self.df= pd.DataFrame(columns=['bonjour'])
        else:
            self.df=load_existing
        self.existing_words = self.df.columns.tolist()

    def _update_table_with_word_(self, ix,word,new=False):
        if new is True:
            self.df[word] = 0
            self.existing_words.append(word)
        self.df.loc[ix, word] += 1

    def _find_possible_existing_words_(self, token, existing_words= None,fuzzy_threshold=0.9, min_string_length=6):
        if existing_words is None:
            existing_words = self.existing_words
        if token in existing_words:
            return {'word':token,'new':False}
        elif len(token) < min_string_length:
            return {'word':token,'new':True}
        else:
            words_score = pd.Series(existing_words,index=existing_words).apply(
                lambda r: ncs.compare_twostrings(token, r, minlength=min_string_length, threshold=fuzzy_threshold))
            if words_score.max() >= fuzzy_threshold:
                word=words_score.argmax()
                score = words_score.max()
                print(token, ' is like ',word,' | score : ',score)
                return {'word': word, 'new': False}
            else:
                return {'word': token, 'new': True}
    def _non_fuzzy_word_(self,word):
        return {'word': word, 'new': (word not in self.existing_words)}
    def _expand_table_with_textfield_(self, ix, textfield):
        fuzzy = False
        self.df.loc[ix]=0
        tokens = ncs.split(textfield)
        if tokens is not None:
            for token in tokens:
                token_clean = ncs.normalizechars(token, remove_s=True)
                if token_clean is not None:
                    if fuzzy is True:
                        result=self._find_possible_existing_words_(token_clean)
                    else:
                        result = self._non_fuzzy_word_(token_clean)
                    self._update_table_with_word_(ix, result['word'], result['new'])



    def expand_table_with_serie(self, textserie):
        start = pd.datetime.now()
        print('starting training shape',self.df.shape)
        for ix in textserie.index:
            if ix not in self.df.index:
                self._expand_table_with_textfield_(ix, textserie.loc[ix])
            else:
                pass
        print('end training shape',self.df.shape)
        end = pd.datetime.now()
        duration = (end - start).total_seconds()
        print('duration time:',duration)

    def word_count(self):
        return self.df.sum(axis=0).sort_values(ascending=False)

    def number_tokens(self):
        return self.df.sum(axis=1)

    def shape(self):
        return self.df.shape

    def pca(self):
        ## to do implement PCA analysis on the X-AXIS to find the words that are commonly co-occuring ##
        pass

    def correct_words_fuzzymatching(self,fuzzy_threshold=0.9,min_string_length=6):
        word_count=self.word_count()
        new_word_count=pd.Series()
        new_df=pd.DataFrame(index=self.df.index,columns=['bonjour'])
        for token in word_count.index:
            result = self._find_possible_existing_words_(token=token,
                                                         existing_words=new_word_count.index.tolist(),
                                                         fuzzy_threshold=fuzzy_threshold,
                                                         min_string_length=min_string_length)
            if result['new']:
                new_word_count.loc[result['word']]=word_count[token]
                new_df[result['word']]=self.df[token]

            else:
                new_word_count.loc[result['word']]+=word_count[token]
                new_df[result['word']] += self.df[token]
        print('previous word count:',word_count.shape[0])
        print('new word count:',new_word_count.shape[0])
        return None
