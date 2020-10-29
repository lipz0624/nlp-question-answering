from question_process import *
from passage_retrieval import *
import re
from nltk.util import ngrams

nlp = spacy.load('en_core_web_sm')
def n_gram(passage,n):
    """
    Input: passage --> string to extract n-grams
    n --> n-grams
    output: list of n-gram lm
    """
    passage = passage.lower()
    passage = re.sub(r'[^a-zA-Z0-9\s]', ' ', passage)
    tokens = [token for token in passage.split(" ") if token != ""]
    output = list(ngrams(tokens, 5))
    return output

def feature(lm, keyQuery, answer):
    """
    Question keywords: the number of question keywords in the n-grams
    """
    count = 0
    for gram in lm:
        if answer in gram:
            for key in keyQuery:
                if key in gram:
                    count +=1
    return count

def rank_answer(passages, keyQuery, answerType):
    """
    Input: a list of relevant passages(string) and question
    Features for ranking candidate answers:
    1. Answer Type Match --> for specific answer type
    2. Pattern match --> for unknown 
    3. add other featuress to improve the performance of answer ranking
    """
    candidates = []
    q_str = ' '.join(keyQuery)
    possible_ans = {} #store possible answers for quantity;key-->answer;value-->score
    for passage in passages:#every passage(string)
        if len(candidates) > 10:
            break
        doc = nlp(passage)
        if answerType == "PERSON":
            for entity in doc.ents:
                if entity.label_ == "PERSON" or entity.label_ == "NORP" or entity.label_ == "ORG":
                    answer = entity.text  
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "LOCATION":
            for entity in doc.ents:
                if entity.label_ == "GPE" or entity.label_ == "LOC" or entity.label_ == "FAC":
                    answer = entity.text
                    lm = n_gram(passage,5) #5-gram lm
                    count = feature(lm, keyQuery, answer)
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        possible_ans[answer] = count
            sort_orders = sorted(possible_ans.items(), key=lambda x: x[1], reverse=True)
            for i in sort_orders:
                candidates.append(i[0])
        elif answerType == "DATE":
            for entity in doc.ents:
                if entity.label_ == "DATE" or entity.label_ == "TIME":
                    answer = entity.text
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "QUANTITY":
            for entity in doc.ents:
                if entity.label_ == "CARDINAL" or entity.label_ == "QUANTITY":
                    answer = entity.text
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "MONEY":
            for entity in doc.ents:
                if entity.label_ == "MONEY" :
                    answer = entity.text
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "UNK":
            T = getChunk(passage)
            for child in T:
                if type(child) == Tree:
                    answer = ' '.join(x[0] for x in  child.leaves())
                    lm = n_gram(passage,5) #5-gram lm
                    count = feature(lm, keyQuery, answer)
                    if answer in q_str: ##we dont want entity occured in the question
                        continue
                    else:
                        possible_ans[answer] = count
            sort_orders = sorted(possible_ans.items(), key=lambda x: x[1], reverse=True)
            for i in sort_orders:
                candidates.append(i[0])
    candidateAnswer = list(OrderedDict.fromkeys(candidates))
    return candidateAnswer[:10]

def writeAns(filename, answers, qid):
    '''write answers to the question (question index - qid) to the file (filename )
    Warning: using append mode so every time need to delete the previous generated file'''
    with open(filename, "a") as f:
        f.write("qid " + str(qid) + '\n')
        for ans in answers:
            f.write(ans + '\n')