from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
# from gensim.models import Word2Vec
import re

prefix = 'hw6_data/training/topdocs/top_docs.' # hardcoded here TODO need change
N = 40 # chunk size
R_SIZE = 10 # the number of retrieved passage

def parse(filename):
  ''' read topdocs file and return a dictionary {docno : text}
  '''
  # TODO encoding?
  docs = {}
  print(filename)
  with open(filename, 'r', encoding='latin-1') as f:
    text = ""
    readmode = False
    for line in f:
      temp = re.findall(r"[A-Z]+\d+-\d{2,}", line)
      if len(temp) > 0:
        docno = temp[0]
        # if (docno == 'AP901217-0185'):
          # print("here", docno)
      elif '<TEXT>' in line:
        readmode = True
      elif '</TEXT>' in line:
        readmode = False
        text = text.replace('\n', ' ')
        # print(docno)
        docs[docno] = text
      elif readmode :
        # TODO find a good way to read, so far: some words are connected
        # text += " ".join(re.findall(r'\w+', line.strip('<P>/')))
        if ('<P>' not in line) and ('</P>' not in line):
          text += line
  # print(len(docs))
  # print(docs['AP901217-0185'])
  return docs


def createCorpus(question, index, switch=False):
  '''id not empty for training where we only read one document and tokenized to 20, 
  false for test where we read 50 docs'''
  filename = prefix + str(index)
  topdocs = parse(filename)
  # print("len of topdocs: ", len(topdocs))
  data_corpus = [' '.join(question)]
  if switch:
    data_corpus += chunk(topdocs[relevant[index]])
  else:
    for doc in topdocs:
      # print(doc)
      data_corpus += chunk(topdocs[doc])

  # print(len(data_corpus))
  return data_corpus

def chunk(doc):
  l = []
  words = doc.split()
  blocks = [words[i:i + N] for i in range(0, len(words), N)]
  for b in blocks:
    l.append(" ".join(b))
  return l

def countFeatureVec(data):
  vectorizer=CountVectorizer()
  X = vectorizer.fit_transform(data)
  a = X.toarray()
  # print(a)
  cos = {}
  for i in range(1, len(a)):
    cos_sim = cosine_similarity([a[0]], [a[i]])
    cos[i] = cos_sim.tolist()[0][0]
  cos_sort = sorted(cos, key=cos.get, reverse=True)
  # print(len(cos_sort))
  ans = []
  i = 0
  while len(ans) != R_SIZE and i < len(cos_sort):
    # print(cos_sort[i])
    # if data[cos_sort[i]] not in ans:
    ans.append(data[cos_sort[i]])
    i += 1
  # print(ans)
  # print(X.toarray())
  return ans

def parseRelevantDocs(filename):
  relevant_docs = {}
  with open(filename, 'r', encoding='utf-8-sig') as f:
    for line in f:
      l = line.strip().split()
      if len(l) > 0 :
        relevant_docs[int(l[0])] = l[1]
  return relevant_docs

# docs_0 = parse(prefix)
relevant = parseRelevantDocs('hw6_data/training/qadata/relevant_docs.txt')
# print(relevant)


