from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser
from nltk.tree import Tree
import string
from nltk.corpus import wordnet,stopwords
import spacy
# from spacy.tokens import Span
from collections import OrderedDict
from passage_retrieval import *
from answer_extraction import *

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
    remove_punc = str.maketrans('', '', string.punctuation)
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        sents = [line.strip().split() for line in file]
        for element in sents:
            if len(element) == 0:
                sents.remove(element)
        for i in range(len(sents)-1):
            if sents[i][0] == 'Number:':
                key = int(sents[i][1])
                value = ' '.join(sents[i+1])
                value = value.translate(remove_punc) #remove punctuation
                questions[key] = value
    return questions
#####################Question Processing####################################
def queryFormulation(nlp,question):
    """
    Based on Keyword Selection Algorithm do the query formulation;
    1. remove stopwords that do not help identifying relevant documents,
        including too common words and purelt functional words 
    2. select all NNP words in recognized named entities
    3. select all nouns with their adjectival modifiers
    4. select all verbs
    """
    key_query = [] #the list recording the key query
    # ##baseline system ==> remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(question)
    word_tokens[0] = word_tokens[0].lower()
    filter_stopWords = [w for w in word_tokens if not w in stop_words]
    sentence = ' '.join(filter_stopWords) ##sentence after removing stop words
    doc = nlp(sentence)
    for entity in doc.ents: ##second step:select all NNP words(named entities)
        for ent in entity.text.split():
            key_query.append(ent)
    for word in doc:
        if word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
            if word.n_lefts > 0:
                for token in word.lefts:
                    #select adjectival modifiers of nouns
                    key_query.append(token.lemma_)
            ##always select noun
            key_query.append(word.lemma_)
        if word.pos_ == "VERB": #select all verbs
            key_query.append(word.lemma_)
    return list(OrderedDict.fromkeys(key_query))

def answerTypeDetection(nlp,question):
    """
    Determine type of expected answer depending of question type
    Type of answer among following --> 
    PERSON(include ORGANIZATION), LOCATION, DATE, QUANTITY, DEFINITION, ABBREV,UNK
    """
    # t_question = "kind of a sports team is the Wisconsin Badgers?"
    q_Tags = ['WP','WDT','WRB'] ##WP ->  who, what WRB -> where, how, when
    qPOS = pos_tag(word_tokenize(question)) #add pos tagging for every word in question
    # print(qPOS)
    qTag = None

    for token in qPOS:
        if token[1] in q_Tags: #if any pos tag is in the question tags
            qTag = token[0].lower()
            break
    # print(qTag)
    if(qTag == None):
        if len(qPOS) > 1:
            if qPOS[0][0].lower() == 'whats':
                qTag = "what"

    if qTag == "who":
        return "PERSON"
    elif qTag == "where":
        return "LOCATION"
    elif qTag == "when":
        return "DATE"
    elif qTag == "what":  
        ##other rules use the headword of the first noun phrase after what
        T = getChunk(question)
        flag_what = False #only become true when after what
        answer = ''
        for child in T:
            if type(child) == tuple:
                if child[1] == 'WP':
                   flag_what = True
            if type(child) == Tree:
                label = child.label()
                # phrase = ' '.join(x[0] for x in  child.leaves())
                if label == "NP" and flag_what == True:
                    answer = child.leaves()[-1][0]
                    break
        
        if answer in ['city','country','state','continent','area','province']:
            return "LOCATION"
        elif answer in ['time','year','date','day']:
            return "DATE"
        elif answer in ['price'] :
            return "QUANTITY"
        return "UNK"
    elif qTag == "how":
        if len(qPOS)>1:
            t2 = qPOS[2]
            if t2[0].lower() in ['many','much']:
                return "QUANTITY"    
        return "UNK"
    else:
        return "UNK"

def getChunk(question):
    qPOS = pos_tag(word_tokenize(question))
    t = ne_chunk(qPOS)
    # print("TTTTT:", t)
    Pattern = "NP:{<DT>?<JJ|PR.>*<NN|NNS>}"
    np_parser = RegexpParser(Pattern)
    T = np_parser.parse(t)
    return T

if __name__ == "__main__":
    # TODO 1. need to clean predict.txt
    nlp = spacy.load('en_core_web_sm')
    questions = read_questions("hw6_data/training/qadata/questions.txt")
    # train_list = []
    for key in questions:
        # train_list.append(questions.get(key))
        question = questions[key]
        q_new = queryFormulation(nlp,question)
        # print(q_new)
        # passage retrieval
        corpus = createCorpus(q_new, key, True)
        retrieved_block = countFeatureVec(corpus)
        # print("Retrieved block\n", retrieved_block)
        # answer
        top_10_ans = rank_answer(retrieved_block, question)
        # print(top_10_ans)
        # print(answerTypeDetection(nlp,question))
        #write ans
        writeAns("predict.txt", top_10_ans, key)