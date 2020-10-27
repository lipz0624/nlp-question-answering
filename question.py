from dataclasses import dataclass
from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser,tree
from nltk.corpus import wordnet,stopwords
import string
import spacy
from spacy.tokens import Span
from collections import OrderedDict

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
    with open(input_file, 'r',encoding='utf-8-sig') as file:
        sents = [line.strip().split() for line in file]
        for element in sents:
            if len(element) == 0:
                sents.remove(element)
        for i in range(len(sents)-1):
            if sents[i][0] == 'Number:':
                value = sents[i][1]
                key = ' '.join(sents[i+1])
                key = key.translate(remove_punc) #remove punctuation
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
    filter_stopWords = [w for w in word_tokens if not w in stop_words]
    sentence = ' '.join(filter_stopWords) ##sentence after removing stop words
    doc = nlp(sentence)
    for entity in doc.ents: ##second step:select all NNP words(named entities)
        # key_query.append((entity.text,entity.label_))
        key_query.append(entity.text)
    for word in doc:
        if word.pos_ == 'NOUN':
            if word.n_lefts > 0:
                for token in word.lefts:
                    #select adjectival modifiers of nouns
                    # key_query.append((token.lemma_,token.pos_))
                    key_query.append(token.lemma_)
            ##always select noun
            # key_query.append((word.lemma_,word.pos_))
            key_query.append(word.lemma_)
        if word.pos_ == "VERB": #select all verbs
            # key_query.append((word.lemma_,word.pos_))
            key_query.append(word.lemma_)
    return list(OrderedDict.fromkeys(key_query))

def answerTypeDetection(nlp,question):
    """
    Determine type of expected answer depending of question type
    Type of answer among following --> 
    PERSON, LOCATION, DATE, ORGANIZATION, QUANTITY, DEFINITION, ENTITY, UNK
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
            if qPOS[0][0].lower() == 'can':
                return "AFFIRMATION"
            elif qPOS[0][0].lower() == 'whats':
                qTag = "what"

    if qTag == "who":
        return "PERSON"
    elif qTag == "where":
        return "LOCATION"
    elif qTag == "when":
        return "DATE"
    elif qTag == "what":
        doc = nlp(question)
        #usually definition problem always start with is/are/was/were
        #so we pay attention to the rest part
        rest_chunk = ' '.join(question.split()[2:])
        if qPOS[1][0] in ['is','are','was','were']:
            rest = nlp(rest_chunk)
            for chunk in rest.noun_chunks:
                if rest_chunk == chunk.text:
                    return "DEFINITION"
        
        ##other rules use the headword of the first noun phrase after what
        # for word in doc:
        #     if word.pos_ == "NOUN":
        #         print('----------')
        #         print(word.text)
        #         print(word.head.text)
        #         if word.head.lemma_ in ['city','country','state','continent']:
        #             return "LOCATION"
        #         elif word.head.lemma_ in ['flower','animal']:
        #             return "ENTITY"
        #         elif word.head.lemma_ in ['name','']:
        #             return "Name"
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
        train_list.append(key)
    question = train_list[0]
    print(question)
    nlp = spacy.load('en_core_web_sm')
    print(queryFormulation(nlp,question))
    print(answerTypeDetection(nlp,question))
    #-----below is example code from spacy API--------
    # print([(w.text, w.pos_) for w in doc])
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # sent = spu(question)
    # for entity in sent.ents:
    #     print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))