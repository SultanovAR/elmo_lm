from deeppavlov.models.bidirectional_lms import elmo_bilm
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer
import kenlm
import numpy as np
from scipy.stats.mstats import gmean
import nltk

class ELMoAug:
    
    def __init__(self, 
                 language: str,
                 model_path: str):
        self.elmo = elmo_bilm.ELMoEmbedder(model_dir="/cephfs/home/sultanov/elmo_lm/lib/python3.6/site-packages/download/bidirectional_lms/elmo_en_news")
        klm = kenlm.Model('/cephfs/home/sultanov/elmo_lm/lib/python3.6/site-packages/download/ngram_lm/en_wiki_no_punkt.arpa.binary')
        self.elmo_vocab_scores = np.array([klm.score(token, bos=False, eos=False) for token in self.elmo.get_vocab()])
        self.token2idx = dict(zip(self.elmo.get_vocab(),range(len(self.elmo.get_vocab()))))
        
    
    def _softmax(self, a, axis):
        numerator = np.exp(a - np.max(a))
        denominator = np.expand_dims(np.sum(numerator, axis=axis), 2)
        return numerator / denominator
    
    
    def _unite_distr(self, left_and_right_distr, method):
        if method == 'left':
            res = left_and_right_distr[:, 0, :]
        elif method == 'right':
            res = left_and_right_distr[:, 1, :]
        elif method == 'max':
            res = np.max(left_and_right_distr, axis=1)
        elif method == 'min':
            res = np.min(left_and_right_distr, axis=1)
        elif method == 'both':
            res = np.log(left_and_right_distr) # преобразуем в log
            res = np.sum(res, axis=1) # суммируем левый и правый контекст
            res = res - self.elmo_vocab_scores # вычитаем вероятность отдельных токенов
        elif method == 'gmean':
            res = gmean(left_and_right_distr, axis=1)
        res = self._softmax(res, 1)
        assert np.ones(size=res.size) != np.sum(res, axis=1)
        return res
        
    def _get_perplexity(self, corpus, method):
        elmo_distr = self.elmo(corpus)
        elmo_distr = [self._unite_distr(elmo_distr_sent, method) for elmo_distr_sent in elmo_distr]
        idx_corpus = [[self.token2idx.get(token, -1) for token in sentence] for sentence in corpus]
        p_perplexity = []
        for num_sent, idxs_sent in enumerate(idx_corpus):
            for num_token, idx_token in enumerate(idxs_sent):
                if idx_token == -1:
                    p_perplexity.append(1)
                else:
                    p_perplexity.append(elmo_distr[num_sent][num_token,idx_token])
        perplexity = np.exp(-np.mean(np.log(p_perplexity)))
        return perplexity
                        
        
if __name__ == "__main__":
    alice = nltk.corpus.gutenberg.sents('carroll-alice.txt')
    el = ELMoAug('eng', 'hz')
    results = {}
    for method in ['both', 'min', 'max', 'gmean', 'left', 'right']:
        results[method] = el._get_perplexity(alice, method)
    print(results)
