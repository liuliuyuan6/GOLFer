import json
import torch 
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy import *
from scipy.stats import entropy


class hyFilter:
    def __init__(self, generation_model_id,nli_model_id,NLI_score=0.85,fact_score=0.85):
        self.generation_model_id=generation_model_id
        self.nli_model_id=nli_model_id #'deberta-v3-large-mnli'
        self.tokenizer= AutoTokenizer.from_pretrained(model_id)
        self.model=AutoModelForCausalLM.from_pretrained(model_id)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("autodl-tmp/en_core_web_sm")
        
        
    def NLIfilter(self, hypothesis_documents, NLI_score=0.85):
        selfcheck_nli = SelfCheckNLI(device=self.device, nli_model=self.nli_model)
        filted_passages=[]
        for l in range(len(hypothesis_documents)):
            passage=hypothesis_documents[l]
            passage =  '"' * 3 + passage + '"' * 3 
            passage=passage.replace("\n", " ").replace("\t", " ").strip()
            sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
            sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]

            hypothesis_documents_left=[x for i, x in enumerate(hypothesis_documents) if i != l]
                

            sent_scores_nli = selfcheck_nli.predict(
                sentences = sentences,                          # list of sentences
                sampled_passages = hypothesis_documents_left, # list of sampled passages
            )
            filted_sentences=[sentence for sentence,sent_score in zip(sentences, sent_scores_nli) if sent_score<NLI_score]
            filted_passage = ''.join(filtedsentences)
        filted_passages.append(filted_passage)
        return filted_passages
  
    def FACTfilter(self, hypothesis_documents,fact_score=0.85):
        filted_passages=[]
        for l in range(len(hypothesis_documents)):
            passage=hypothesis_documents[l]
            passage =  "\"" * 3 + passage + "\"" * 3
            passage=passage.replace("\n", " ").strip()
            sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
            sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
            filted_sentences=[]
            for i in range(len(sentences)):
                input = self.tokenizer(sentences[i], return_tensors="pt").to(self.device)
                output1=self.model(input.input_ids,output_attentions=True)
                logits = output1.logits
                prob = torch.softmax(logits, dim=-1)[0]
                entropies=entropy(prob.cpu().detach().numpy(), base=2,axis=-1)
                attentions=output1.attentions
                attentions = attentions[-1][0]
                mean_atten = torch.sum(attentions, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for k in range(mean_atten.shape[0]):
                    mean_atten[k] /= (mean_atten.shape[0] - k)
                mean_atten=mean_atten.cpu().detach().numpy()
                sen_entropyAtten=entropies[1:]@mean_atten[1:]/len(mean_atten[1:])
                if sen_entropyAtten<fact_score:
                    filted_sentences.append(sentences[i])
            filted_passage = ''.join(filtedsentences)
        filted_passages.append(filted_passage)
        return filted_passages

    def hyfilt(self, hypothesis_documents,NLI_score=0.85,fact_score=0.85):
        selfcheck_nli = SelfCheckNLI(device=self.device, nli_model=self.nli_model)
        filted_passages=[]
        for l in range(len(hypothesis_documents)):
            sents_probs={}
            passage=hypothesis_documents[l]
            passage =  '"' * 3 + passage + '"' * 3 
            passage=passage.replace("\n", " ").replace("\t", " ").strip()
            sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
            sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]

            hypothesis_documents_left=[x for i, x in enumerate(hypothesis_documents) if i != l]
                

            sent_scores_nli = selfcheck_nli.predict(
                sentences = sentences,                          # list of sentences
                sampled_passages = hypothesis_documents_left, # list of sampled passages
            )
            filted_sentences_NLI=[sentence for sentence,sent_score in zip(sentences, sent_scores_nli) if sent_score<NLI_score]
            filted_sentences_fact=[]
            for i in range(len(sentences)):
                input = self.tokenizer(sentences[i], return_tensors="pt").to(self.device)
                output1=self.model(input.input_ids,output_attentions=True)
                logits = output1.logits
                prob = torch.softmax(logits, dim=-1)[0]
                probcpu=prob.cpu().detach().numpy()
                entropies=entropy(prob.cpu().detach().numpy(), base=2,axis=-1)
                attentions=output1.attentions
                attentions = attentions[-1][0]
                mean_atten = torch.sum(attentions, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for k in range(mean_atten.shape[0]):
                    mean_atten[k] /= (mean_atten.shape[0] - k)
                mean_atten=mean_atten.cpu().detach().numpy()
                sent_entropyAtten=entropies[1:]@mean_atten[1:]/len(mean_atten[1:])
                sent_probs=[]
                for k in range(input.input_ids.size()[1]-1):
                    sent_probs.append(probcpu[k+1,input.input_ids[0][k+1]].astype(float))
                sents_probs[sentences[i]]=sent_probs
                if sent_entropyAtten<fact_score:
                    filted_sentences_fact.append(sentences[i])
            filted_sentences=list(set(filted_sentences_NLI) & set(filted_sentences_fact))
            num_tokens=0
            tokens_probs=0
            for p in range(len(filted_sentences)):
                num_tokens+=len(sents_probs[filted_sentences[p]])
                for q in range(len(sents_probs[filted_sentences[p]])):
                    tokens_probs+=sents_probs[filted_sentences[p]][q]
            if num_tokens!=0:
                passage_prob=tokens_probs/num_tokens
            else:
                passage_prob=0
            filted_passage = ''.join(filtedsentences)
            filted_passages.append([filted_passage,passage_prob])
        return filted_passages
  
