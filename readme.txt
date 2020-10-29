Paige Li, Yunqi Shen
HW 6 - NLP

To compile and run use: "python3 question_processing.py" 
Then it will give you prompt to choose whether you are going to train files mode or test files mode.
Train -> enter 'y'; Test -> enter 'n'
Then it will generate predict.txt that contains the answers in desired format.
###Warning: if predict.txt need to be clear everytime you rerun the programs.  

Dependencies:
1. ntlk
    - from nltk.util import ngrams
    - from nltk.stem import WordNetLemmatizer
    - from nltk.corpus import wordnet,stopwords
    - from nltk import word_tokenize,pos_tag,ne_chunk,RegexpParser
    - from nltk.tree import Tree
2. Spacy - (pip install -U spacy) or pip3
3. scikit learn (pip install -U scikit-learn) or pip3 
4. NumPy
