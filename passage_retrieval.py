from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk import sent_tokenize
import re
import time
prefix = 'hw6_data/training/topdocs/top_docs.' # train
# prefix = 'hw6_data/test/topdocs/top_docs.'  # test
N = 30 # chunk size
R_SIZE = 20 # the number of retrieved passage

def parse(filename):
  ''' read topdocs file and return a dictionary {docno : text}
  '''
  docs = {}
  with open(filename, 'r', encoding='latin-1') as f:
    text = ""
    readmode = False
    for line in f:
      temp = re.findall(r"[A-Z]+\d+-\d{2,}", line) # find docno code
      if len(temp) > 0:
        docno = temp[0]
      elif '<TEXT>' in line:
        readmode = True
      elif '</TEXT>' in line:
        readmode = False
        text = text.replace('\n', ' ')
        docs[docno] = text
      elif readmode :
        if ('<P>' not in line) and ('</P>' not in line):
          # text += " ".join(re.findall(r'\w+', line.strip('<P>/'))) # get rid of puctuation
          text += line
  return docs


def createCorpus(question, index, switch=False):
  '''
  @param 
  question: keyquery after question processing
  index: question number
  switch: True only for testing training data
  '''
  filename = prefix + str(index)
  topdocs = parse(filename)
  data_corpus = [' '.join(question)]
  if switch:
    # data_corpus += chunk(topdocs[relevant[index]])
    data_corpus += sent_tokenize(topdocs[relevant[index]])
    # data_corpus += sent(topdocs[relevant[index]])
  else:
    for doc in topdocs:
      # data_corpus += chunk(topdocs[doc])
      data_corpus += sent_tokenize(topdocs[doc])
      # data_corpus += sent(topdocs[doc])
  return data_corpus

def chunk(doc):
  '''helper method to split a string into a list of chunks. (N tokens per chunk)'''
  l = []
  words = doc.split()
  blocks = [words[i:i + N] for i in range(0, len(words), N)]
  for b in blocks:
    l.append(" ".join(b))
  return l

def passageRetrieve(data):
  '''
  @param data: a list of string chunks, question is the first one
  '''

  # generate feature vector
  # start_time = time.time()
  vectorizer = CountVectorizer(stop_words='english', binary=True)
  X = vectorizer.fit_transform(data)
  a = X.toarray()
  # cosine similarity
  cos = {}
  for i in range(1, len(a)):
    # cos_sim = 0.0
    # denominator = np.linalg.norm(a[0]) * np.linalg.norm(a[i])
    # if denominator > 0:
    #   cos_sim = np.dot(a[0], a[i]) / denominator
    # cos[i] = cos_sim
    # cos_sim = cosine_similarity([a[0]], [a[i]])
    # cos[i] = cos_sim.tolist()[0][0]
    cos[i] = np.dot(a[0], a[i])
  cos_sort = sorted(cos, key=cos.get, reverse=True)
  # return top passages
  ans = []
  i = 0
  while len(ans) != R_SIZE and i < len(cos_sort):
    if data[cos_sort[i]] not in ans:
      ans.append(data[cos_sort[i]])
    i += 1
  # elapsed_time = time.time() - start_time
  # print(' return Took {:.03f} seconds'.format(elapsed_time))
  return ans

def parseRelevantDocs(filename):
  relevant_docs = {}
  with open(filename, 'r', encoding='utf-8-sig') as f:
    for line in f:
      l = line.strip().split()
      if len(l) > 0 :
        relevant_docs[int(l[0])] = l[1]
  return relevant_docs

relevant = parseRelevantDocs('hw6_data/training/qadata/relevant_docs.txt')



