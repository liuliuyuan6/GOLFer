import json
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder,AnceQueryEncoder
import numpy as np
import random



class SparseSearcher():
    def __init__(self, searcher_name, ratio=0.75,k=1000):
        self.searcher_name=searcher_name
        self.ratio = ratio
        self.k=k

    def sparsesearch(query,hypothesis_documents):
        searcher = LuceneSearcher.from_prebuilt_index(self.searcher_name)
        coe=ratio*5*len(hypothesis_documents)
        query=query*coe
        hypothesis_documents=''.join(hypothesis_documents)
        hits=self.searcher.search(query+hypothesis_documents,k=self.k)
        return hits
    
class DenseSearcher():
    def __init__(self, searcher_name, encoder_dir,ratio1=0.075,ratio2=0.25,max_tokens=128,k=1000):
        self.searcher_name=searcher_name
        self.encoder=encoder
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.k=k
        self.coe=ratio1*max_tokens+ratio2
        
    def densesearch(query,hypothesis_documents_withprobs):
        encoder = AutoQueryEncoder(self.encoder_dir, pooling='mean')
        searcher = FaissSearcher.from_prebuilt_index(self.searcher_name, encoder)

        coe_passages=(1-self.coe)/np.sum([[row[1]] for row in hypothesis_documents_withprobs])
        prob=[[self.coe]]+[[row[1]*coe_passages] for row in hypothesis_documents_withprobs]
        hypothesis_documents=[row[0] for row in hypothesis_documents_withprobs]

        all_emb_c = []
        for hypothesis_document in [query]+hypothesis_documents:
            c=hypothesis_document
            c_emb = encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        weighted_emb_c = np.sum(prob*all_emb_c, axis=0)
        GOLFer_vector = weighted_emb_c.reshape((1, len(weighted_emb_c)))
        hits=self.searcher.search(GOLFer_vector,k=self.k)
        return hits
        
