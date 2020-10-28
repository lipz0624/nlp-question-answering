from question_process import *
from passage_retrieval import *
from date_extractor import extract_date

def rank_answer(passages,question):
    """
    Input: a list of relevant passages(string) and question
    Features for ranking candidate answers:
    1. Answer Type Match
    2. Pattern match
    3.# of question keywords in the candidate
    """
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
                    q_str = ' '.join(keyQuery)
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    # ##TODO: answer should not be only one word
                    # and if there are multiple candidate answers, need ranking
        elif answerType == "LOCATION":
            for entity in doc.ents:
                if entity.label_ == "GPE" or entity.label_ == "LOC" or entity.label_ == "FAC":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    q_str = ' '.join(keyQuery)
                    if answer in q_str:##we dont want entity occured in the question
                        continue
        # elif answerType == "DATE":
            # allDates = []
            # allDates.extend(extractDate())
            # if len(allDates)>0:
            #     answer = allDates[0]

    return answer



nlp = spacy.load('en_core_web_sm')
text = "what Paige is at hotel yesterday night 12:30 at $2, Janurary 1937"
# answerTypeDetection(nlp, text)
doc = nlp(text)
print("ENTITY:", doc.ents)
for a in doc.ents:
    print(a.label_)
date = extract_date(text)
print("DATE:", date)
# print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-'])