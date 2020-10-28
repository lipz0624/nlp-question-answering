from question_process import *
from passage_retrieval import *

def rank_answer(passages,question):
    """
    Input: a list of relevant passages(string) and question
    Features for ranking candidate answers:
    1. Answer Type Match --> for specific answer type
    2. Pattern match --> for unknown 
    3. add other featuress to improve the performance of answer ranking
    """
    candidates = []
    ##expected answer type
    answerType = answerTypeDetection(nlp,question)
    keyQuery = queryFormulation(nlp,question)
    for passage in passages:#every passage(string)
        doc = nlp(passage)
        if answerType == "PERSON":
            for entity in doc.ents:
                if entity.label_ == "PERSON" or entity.label_ == "NORP" or entity.label_ == "ORG":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    q_str = ' '.join(keyQuery)
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "LOCATION":
            for entity in doc.ents:
                if entity.label_ == "GPE" or entity.label_ == "LOC" or entity.label_ == "FAC":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    q_str = ' '.join(keyQuery)
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "DATE":
            ##TODO: extract time in passages
            answer = -1
            candidates.append(answer)
        elif answerType == "UNK":
            T = getChunk(question)
            for child in T:
                if type(child) == Tree:
                    label = child.label()
                    if label == "NP":
                        answer = child.leaves()[-1][0]
                        if answer in q_str:##we dont want entity occured in the question
                            continue
                        else:
                            candidates.append(answer)
        ##TODO: whether add definition as an answerType because it 
        # should return the whole sentence inteand of just one word
    candidateAnswer = list(OrderedDict.fromkeys(candidates))
    return candidateAnswer[:10]