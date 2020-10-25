from dataclasses import dataclass
from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser,tree
from nltk.corpus import wordnet,stopwords
import string

# @dataclass
# class Question:
#     """Class for keeping track of an question's information."""
#     number: int
#     query: str
#     questionType: str
#     answerType: str
#     focus: str

def read_questions(input_file):
    """
    read and record every question in the input file and record its number as a value
    so in dict questions --> Key = question(str) Value = number(int)
    """
    questions = {}
    with open(input_file, 'r',encoding='utf-8-sig') as f:
        sents = [line.strip().split() for line in f]
        for element in sents:
            if len(element) == 0:
                sents.remove(element)
        for i in range(len(sents)-1):
            if sents[i][0] == 'Number:':
                value = sents[i][1]
                key = ' '.join(sents[i+1])
                questions[key] = value
    return questions

def findQueType(question):
    """
    To determine type of question by analyzing POS tag of question
    """
    # WP -> who, WDT -> what, why, how, WP$ -> whose , WRB -> where
    questionTags = ['WP','WDT','WP$','WRB']
    qpos_tag = pos_tag(word_tokenize(question))
    qTags = []
    ## special case: name a ...
    for pair in qpos_tag:
        if pair[1] in questionTags:
            qTags.append(pair[1])
    queType = ''
    if(len(qTags) == 1):
        queType = qTags[0]
    elif(len(qTags) > 1):
        queType = 'Multi'
    else:
        if word_tokenize(question)[0] == 'Name':
            queType = 'List'
        else:
            queType = 'None'
    return queType

# def findAnsType(question):
#     """
#     Determine the answer type of the question
#     """
#     questionTags = ['WP','WDT','WP$','WRB']
#     q = question.lower()
#     qPOS = pos_tag(word_tokenize(q))
#     qTag = None
#     for pair in qPOS:
#         if pair[1] in questionTags:
#             qTag = pair[0]
#             break

#     if qTag == 'where':
#         return 'Location'
#     elif qTag == 'who':
#         return 'Person'
#     elif qTag == 'when':
#         return 'Date'
#     elif qTag == 'what':
#         ####DIFFICULT TO DISTINGUISH
#         for pair in qPOS:
#             if pair[0] in ['is','are','was','were',"'s"]:
#                 return 'Definition'
#             elif pair[0] in ['city','area','state','continent','province']:
#                 return 'Location'
#             elif 'year'  in q:
#                 return 'Date'
#             else: #otherwise, we are not sure about answer type
#                 return 'UNK'
#     else:
#         return 'None'

def formQuery(question):
    """
    Based on keyword Selection Algorithm do the query formulation;
    1. select all non-stop words
    2. select all NNP words in recognized named entities
    """
    query = []
    remove_punc = str.maketrans('', '', string.punctuation)
    ##baseline system ==> remove stop words
    stop_words = set(stopwords.words('english')) 
    question = question.strip().translate(remove_punc)
    word_tokens = word_tokenize(question)
    ## always lower the first word --- NOT SURE
    word_tokens[0] = word_tokens[0].lower()
    filter_stopWords = [w for w in word_tokens if not w in stop_words]
    chunkTree = ne_chunk(pos_tag(filter_stopWords))
    #-----need fix to contain all NNP words and all complex nominal------
    # pattern= "NP: {<DT>?<JJ|PR.>*<NN|NNS>}"
    # np_parser = RegexpParser(pattern)
    # T2 = np_parser.parse(chunkTree)
    # for child in T2:
    #     if type(child) == tree.Tree:
    #         entities = ' '.join(x[0] for x in child.leaves())
    #         query.append(entities)
    return filter_stopWords


if __name__ == "__main__":
    questions = read_questions("hw6_data/training/qadata/questions.txt")
    train_list = []
    for key in questions:
        train_list.append(key)
    question = train_list[0]
    number = questions[question]
    print(formQuery(question))
    # print(number)
    # print(findQueType(question))
    # print(findAnsType(question))