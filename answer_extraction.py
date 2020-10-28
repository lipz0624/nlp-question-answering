from question_process import *
from passage_retrieval import *

def add_score(answer,rank):
    if answer not in rank:
        rank[answer] = {'score':1}
    else:
        rank[answer]['score'] += 1
    return rank

def rank_answer(passages,question):
    """
    Input: a list of relevant passages(string) and question
    Features for ranking candidate answers:
    1. Answer Type Match
    2. Pattern match
    3.# of question keywords in the candidate
    """
    ##dict to store possible answers and its score
    rank = {}
    ##expected answer type
    answerType = answerTypeDetection(nlp,question)
    keyQuery = queryFormulation(nlp,question)
    for passage in passages:
        doc = nlp(passage)
        if answerType == "PERSON":
            for entity in doc.ents:
                if entity.label_ == "PERSON" or entity.label_ == "NORP" or entity.label_ == "ORG":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    ##TODO: answer should not be only one word
                    rank = add_score(answer,rank)
                    ##TODO :regular expression pattern matches the candidate
                    for word in doc: ##features: number of question keywords in the candidate
                        if word.lemma_ in keyQuery:
                            rank = add_score(answer,rank)
        elif answerType == "LOCATION":
            for entity in doc.ents:
                if entity.label_ == "GPE" or entity.label_ == "LOC" or entity.label_ == "FAC":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    rank = add_score(answer,rank)
                    for word in doc: ##features: number of question keywords in the candidate
                        if word.lemma_ in keyQuery:
                            rank = add_score(answer,rank)
    return -1