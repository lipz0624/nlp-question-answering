from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser
from nltk.tree import Tree
import string
from nltk.corpus import wordnet,stopwords
import spacy
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from passage_retrieval import *
from answer_extraction import *
import time

wnl = WordNetLemmatizer()

def read_questions(input_file):
    """
    read and record every question in the input file and record its number as a value
    so in dict questions --> Key = question(str) Value = number(int)
    """
    questions = {}
    remove_punc = str.maketrans('', '', string.punctuation)
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        sents = [line.strip().split() for line in file]
    for i in range(len(sents)-1):
        if len(sents[i]) == 0:
            continue
        elif sents[i][0] == 'Number:':
            key = int(sents[i][1])
            value = ' '.join(sents[i+1])
            value = value.translate(remove_punc) #remove punctuation
            questions[key] = value
    return questions

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
        key_query += entity.text.split()
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
    PERSON(include ORGANIZATION), LOCATION, DATE, QUANTITY, MONEY, UNK
    """
    q_Tags = ['WP','WDT','WRB'] ##WP ->  who, what WRB -> where, how, when
    qPOS = pos_tag(word_tokenize(question)) #add pos tagging for every word in question
    qTag = None

    for token in qPOS:
        if token[1] in q_Tags: #if any pos tag is in the question tags
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
        ##other rules use the headword of the first noun phrase after what
        T = getChunk(question)
        flag_what = False #only become true when after what
        answer = ''
        for child in T:
            if type(child) == tuple:
                if child[1] in q_Tags:
                   flag_what = True
            if type(child) == Tree:
                label = child.label()
                if label == "NP" and flag_what == True:
                    answer = child.leaves()[-1][0]
                    answer = wnl.lemmatize(answer) #lemmatize the string
                    break
        
        if answer in ['city','country','state','continent','area','province']:
            return "LOCATION"
        elif answer in ['king','nationality']:
            return "PERSON"
        elif answer in ['time','year','date','day']:
            return "DATE"
        elif answer in ['population'] :
            return "QUANTITY"
        elif answer in ['price', 'salary']:
            return "MONEY"
        return "UNK"
    elif qTag == "how":
        if len(qPOS)>1:
            t2 = qPOS[1]
            if t2[0].lower() in ['many']:
                return "QUANTITY"
            elif t2[0].lower() in ['much']:
                return "MONEY"        
        return "UNK"
    else:
        return "UNK"

def getChunk(question):
    qPOS = pos_tag(word_tokenize(question))
    t = ne_chunk(qPOS)
    Pattern = "NP:{<DT>?<JJ|PR.>*<NN|NNS>}"
    np_parser = RegexpParser(Pattern)
    T = np_parser.parse(t)
    return T

if __name__ == "__main__":
    # TODO 1. need to clean predict.txt
    nlp = spacy.load('en_core_web_sm')
    train_filename = "hw6_data/training/qadata/questions.txt"
    test_filename = "hw6_data/test/qadata/questions.txt"
    questions = read_questions(train_filename)
    for key in questions:
        question = questions[key]
        # question processing
        q_new = queryFormulation(nlp,question)
        ans_type = answerTypeDetection(nlp, question)
        print(key, question, ans_type)
        # passage retrieval
        corpus = createCorpus(q_new, key, False)
        retrieved_block = passageRetrieve(corpus)
        # answer
        top_10_ans = rank_answer(retrieved_block, q_new, ans_type)
        #write ans
        writeAns("predict.txt", top_10_ans, key)