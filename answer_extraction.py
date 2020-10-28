from question_process import *
from passage_retrieval import *
# from date_extractor import extract_date

nlp = spacy.load('en_core_web_sm')

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
    print(question, answerType)
    keyQuery = queryFormulation(nlp,question)
    q_str = ' '.join(keyQuery)
    for passage in passages:#every passage(string)
        if len(candidates) == 10:
            break
        doc = nlp(passage)
        if answerType == "PERSON":
            for entity in doc.ents:
                if entity.label_ == "PERSON" or entity.label_ == "NORP" or entity.label_ == "ORG":
                    #We want to include all these NE as possible answers
                    answer = entity.text  
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "LOCATION":
            for entity in doc.ents:
                if entity.label_ == "GPE" or entity.label_ == "LOC" or entity.label_ == "FAC":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "DATE":
            for entity in doc.ents:
                if entity.label_ == "DATE" or entity.label_ == "TIME":
                    answer = entity.text
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "QUANTITY":
            for entity in doc.ents:
                if entity.label_ == "MONEY" or entity.label_ == "CARDINAL" or entity.label_ == "QUANTITY":
                    #We want to include all these NE as possible answers
                    answer = entity.text
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
        elif answerType == "UNK":
            T = getChunk(passage)
            # print(T)
            for child in T:
                # print(type(child))
                if type(child) == Tree:
                    # print(label)
                    # print(child.leaves())
                    answer = ' '.join(x[0] for x in  child.leaves())
                    # print('np:' + answer)
                    if answer in q_str:##we dont want entity occured in the question
                        continue
                    else:
                        candidates.append(answer)
    candidateAnswer = list(OrderedDict.fromkeys(candidates))
    return candidateAnswer[:10]


def writeAns(filename, answers, qid):
    with open(filename, "a") as f:
        f.write("qid " + str(qid) + '\n')
        for ans in answers:
            f.write(ans + '\n')


# passages = ['How many team did Wayne Gretzky play for zip at 11?']
# question = 'How many is hockey team?'
# answer = rank_answer(passages,question)
# print(answer)

# nlp = spacy.load('en_core_web_sm')
# text = "what Paige is at hotel yesterday night 12:30 at $2, Janurary 1937, there are two apples and 7 bananas"
# answerTypeDetection(nlp, question)
# doc = nlp(text)
# print("ENTITY:", doc.ents)
# for a in doc.ents:
#     print(a.label_)