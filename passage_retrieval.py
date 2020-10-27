from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from gensim.models import Word2Vec
import re

prefix = 'hw6_data/training/topdocs/top_docs.0' # hardcoded here TODO need change
N = 20 # chunk size

def parse(filename):
  ''' read topdocs file and return a dictionary {docno : text}
  '''
  # TODO encoding?
  docs = {}
  with open(filename) as f:
    text = ""
    readmode = False
    for line in f:
      if '<DOCNO>' in line:
        l = line.split()
        docno = l[1]
        # print(docno)
      elif '<TEXT>' in line:
        readmode = True
      elif '</TEXT>' in line:
        readmode = False
        text = text.replace('\n', ' ')
        docs[docno] = text
      elif readmode :
        # TODO find a good way to read, so far: some words are connected
        text += " ".join(re.findall(r'\w+', line.strip('<P>/')))
        # text += line.strip('<P>/.,')

  # print(docs['LA080989-0132'])
  return docs

def createCorpus(question, topdocs, id=''):
  '''id not empty for training where we only read one document and tokenized to 20, 
  false for test where we read 50 docs'''
  data_corpus = [' '.join(question)]
  if id != '' :
    data_corpus += chunk(topdocs[id])
  else:
    for doc in top_docs:
      data_corpus += chunk(doc)

  # print(data_corpus)
  return data_corpus

def chunk(doc):
  l = []
  words = word_tokenize(doc)
  blocks = [words[i:i + N] for i in range(0, len(words), N)]
  for b in blocks:
    l.append(" ".join(b))
  return l

def countFeatureVec(data):
  vectorizer=CountVectorizer()
  vocabulary=vectorizer.fit(data)
  X = vectorizer.transform(data)
  a = X.toarray()
  # print(a)
  cos = {}
  for i in range(1, len(a)):
    cos_sim = cosine_similarity([a[0]], [a[i]])
    cos[i] = cos_sim.tolist()[0][0]
  # print(type(cos[1]))
  cos_sort = sorted(cos, key=cos.get, reverse=True)
  ans = data[cos_sort[0]]
  # print(ans)
  # print(X.toarray())
  # print(vocabulary.get_feature_names())
  return ans



docs_0 = parse(prefix)