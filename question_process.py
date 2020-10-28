# from dataclasses import dataclass
from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser
from nltk.tree import Tree
from nltk.corpus import wordnet,stopwords
import string  # TODO why import string?
import spacy
from spacy.tokens import Span
from collections import OrderedDict
from passage_retrieval import *

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
                key = sents[i][1]
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
    q_Tags = ['WP','WDT','WRB'] ##WP ->  who WDT -> what, why, how, WRB -> where 
    qPOS = pos_tag(word_tokenize(question)) #add pos tagging for every word in question
    qTag = None

    for token in qPOS:
        if token[1] in q_Tags:#if any pos tag is in the question tags
            qTag = token[0].lower()
            break
    
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
        # doc = nlp(question)
        #usually definition problem always start with is/are/was/were
        #so we pay attention to the rest part
        rest_chunk = ' '.join(question.split()[2:])
        if ' '.join(question.split()[-2:]) == 'stand for?':
            return "ABBREV"
        if qPOS[1][0] in ['is','are','was','were']:
            rest = nlp(rest_chunk)
            for chunk in rest.noun_chunks:
                if rest_chunk == chunk.text:
                    return "DEFINITION"
        
        ##other rules use the headword of the first noun phrase after what
        t = ne_chunk(qPOS)
        flag_what = False #only become true when after what
        Pattern = "NP: {<DT>?<JJ|PR.>*<NN|NNS>}"
        np_parser = RegexpParser(Pattern)
        T = np_parser.parse(t)
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
        elif answer in ['abbreviation']:
            return "ABBREV"
        else:
            return "UNK"
        return "UNK"
    elif qTag == "how":
        if len(qPOS)>1:
            t2 = qPOS[2]
            if t2[0].lower() in ['many','much']:
                return "QUANTITY"    
        return "UNK"
    else:
        return "UNK"

if __name__ == "__main__":
    questions = read_questions("hw6_data/training/qadata/questions.txt")
    train_list = []
    for key in questions:
        train_list.append(questions.get(key))
    question = train_list[0]
    # print(question)
    nlp = spacy.load('en_core_web_sm')
    #question process
    q_new = queryFormulation(nlp,question)
    print(q_new)
    # passage retrieval
    corpus = createCorpus(q_new, docs_0, 'LA080989-0132')
    retrieved_block = countFeatureVec(corpus)
    print(retrieved_block)
    # answer
    print(answerTypeDetection(nlp,question))