from flask import Flask, render_template, request, url_for, jsonify
import nltk
nltk.download('punkt', quiet=True)
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True )
import nltk
nltk.download('wordnet', quiet=True)
app = Flask(__name__)

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import pickle as cPickle
from pathlib import Path
import pandas as pd

def dumpPickle(fileName, content):
    pickleFile = open(fileName, 'wb')
    cPickle.dump(content, pickleFile, -1)
    pickleFile.close()

def loadPickle(fileName):    
    file = open(fileName, 'rb')
    content = cPickle.load(file)
    file.close()
    
    return content
    
def pickleExists(fileName):
    file = Path(fileName)
    
    if file.is_file():
        return True
    
    return False

#Extract answers and the sentence they are in
def extractAnswers(qas, doc):
    answers = []

    senStart = 0
    senId = 0

    for sentence in doc.sents:
        senLen = len(sentence.text)

        for answer in qas:
            answerStart = answer['answers'][0]['answer_start']

            if (answerStart >= senStart and answerStart < (senStart + senLen)):
                answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

        senStart += senLen
        senId += 1
    
    return answers

#TODO - Clean answers from stopwords?
def tokenIsAnswer(token, sentenceId, answers):
    for i in range(len(answers)):
        if (answers[i]['sentenceId'] == sentenceId):
            if (answers[i]['text'] == token):
                return True
    return False

#Save named entities start points

def getNEStartIndexs(doc):
    neStarts = {}
    for ne in doc.ents:
        neStarts[ne.start] = ne
        
    return neStarts 

def getSentenceStartIndexes(doc):
    senStarts = []
    
    for sentence in doc.sents:
        senStarts.append(sentence[0].i)
    
    return senStarts
    
def getSentenceForWordPosition(wordPos, senStarts):
    for i in range(1, len(senStarts)):
        if (wordPos < senStarts[i]):
            return i - 1
        
def addWordsForParagrapgh(newWords, text):
    doc = nlp(text)
    neStarts = getNEStartIndexs(doc)
    senStarts = getSentenceStartIndexes(doc)
    
    #index of word in spacy doc text
    i = 0
    
    while (i < len(doc)):
        #If the token is a start of a Named Entity, add it and push to index to end of the NE
        if (i in neStarts):
            word = neStarts[i]
            #add word
            currentSentence = getSentenceForWordPosition(word.start, senStarts)
            wordLen = word.end - word.start
            shape = ''
            for wordIndex in range(word.start, word.end):
                shape += (' ' + doc[wordIndex].shape_)

            newWords.append([word.text,
                            0,
                            0,
                            currentSentence,
                            wordLen,
                            word.label_,
                            None,
                            None,
                            None,
                            shape])
            i = neStarts[i].end - 1
        #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
        else:
            if (doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]

                currentSentence = getSentenceForWordPosition(i, senStarts)
                wordLen = 1
                
                newWords.append([word.text,
                                0,
                                0,
                                currentSentence,
                                wordLen,
                                None,
                                word.pos_,
                                word.tag_,
                                word.dep_,
                                word.shape_])
        i += 1

def oneHotEncodeColumns(df):
    columnsToEncode = ['NER', 'POS', "TAG", 'DEP']

    for column in columnsToEncode:
        one_hot = pd.get_dummies(df[column])
        one_hot = one_hot.add_prefix(column + '_')
        df = df.drop(column, axis = 1)
        df = df.join(one_hot)
    
    return df


# ## *Predict whether a word is a keyword* 

def generateDf(text):
    words = []
    addWordsForParagrapgh(words, text)

    wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
    df = pd.DataFrame(words, columns=wordColums)

    return df


def prepareDf(df):
    #One-hot encoding
    wordsDf = oneHotEncodeColumns(df)

    #Drop unused columns
    columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape']
    wordsDf = wordsDf.drop(columnsToDrop, axis = 1)

    #Add missing colums 
    predictorColumns = ['wordCount','NER_CARDINAL','NER_DATE','NER_EVENT','NER_FAC','NER_GPE','NER_LANGUAGE','NER_LAW','NER_LOC','NER_MONEY','NER_NORP','NER_ORDINAL','NER_ORG','NER_PERCENT','NER_PERSON','NER_PRODUCT','NER_QUANTITY','NER_TIME','NER_WORK_OF_ART','POS_ADJ','POS_ADP','POS_ADV','POS_CCONJ','POS_DET','POS_INTJ','POS_NOUN','POS_NUM','POS_PART','POS_PRON','POS_PROPN','POS_PUNCT','POS_SYM','POS_VERB','POS_X','TAG_''','TAG_-LRB-','TAG_.','TAG_ADD','TAG_AFX','TAG_CC','TAG_CD','TAG_DT','TAG_EX','TAG_FW','TAG_IN','TAG_JJ','TAG_JJR','TAG_JJS','TAG_LS','TAG_MD','TAG_NFP','TAG_NN','TAG_NNP','TAG_NNPS','TAG_NNS','TAG_PDT','TAG_POS','TAG_PRP','TAG_PRP$','TAG_RB','TAG_RBR','TAG_RBS','TAG_RP','TAG_SYM','TAG_TO','TAG_UH','TAG_VB','TAG_VBD','TAG_VBG','TAG_VBN','TAG_VBP','TAG_VBZ','TAG_WDT','TAG_WP','TAG_WRB','TAG_XX','DEP_ROOT','DEP_acl','DEP_acomp','DEP_advcl','DEP_advmod','DEP_agent','DEP_amod','DEP_appos','DEP_attr','DEP_aux','DEP_auxpass','DEP_case','DEP_cc','DEP_ccomp','DEP_compound','DEP_conj','DEP_csubj','DEP_csubjpass','DEP_dative','DEP_dep','DEP_det','DEP_dobj','DEP_expl','DEP_intj','DEP_mark','DEP_meta','DEP_neg','DEP_nmod','DEP_npadvmod','DEP_nsubj','DEP_nsubjpass','DEP_nummod','DEP_oprd','DEP_parataxis','DEP_pcomp','DEP_pobj','DEP_poss','DEP_preconj','DEP_predet','DEP_prep','DEP_prt','DEP_punct','DEP_quantmod','DEP_relcl','DEP_xcomp']

    for feature in predictorColumns:
        if feature not in wordsDf.columns:
            wordsDf[feature] = 0

    return wordsDf

def predictWords(wordsDf, df):
    
    predictorPickleName = 'C:/Users/abhis/Desktop/Final Year Project/data/nb-predictor.pkl'
    predictor = loadPickle(predictorPickleName)
    
    y_pred = predictor.predict_proba(wordsDf)

    labeledAnswers = []
    for i in range(len(y_pred)):
        labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})

    return labeledAnswers


# ## *Extract questions*
def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
    leftPartStart = doc[sentStart].idx
    leftPartEnd = doc[firstTokenIndex].idx
    rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex])
    rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
    
    question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]
    
    return question



def addQuestions(answers, text):
    doc = nlp(text)
    currAnswerIndex = 0
    qaPair = []

    #Check wheter each token is the next answer
    for sent in doc.sents:
        for token in sent:
            
            #If all the answers have been found, stop looking
            if currAnswerIndex >= len(answers):
                break
            
            #In the case where the answer is consisted of more than one token, check the following tokens as well.
            answerDoc = nlp(answers[currAnswerIndex]['word'])
            answerIsFound = True
            
            for j in range(len(answerDoc)):
                if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
                    answerIsFound = False
           
            #If the current token is corresponding with the answer, add it 
            if answerIsFound:
                question = blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)
                
                qaPair.append({'question' : question, 'answer': answers[currAnswerIndex]['word'], 'prob': answers[currAnswerIndex]['prob']})
                
                currAnswerIndex += 1
                
    return qaPair



def sortAnswers(qaPairs):
    orderedQaPairs = sorted(qaPairs, key=lambda qaPair: qaPair['prob'])
    
    return orderedQaPairs    


# ## *Distractors*
# Taken from the *04. Generating incorrect answers/Incorrect-answers* notebook.


import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

glove_file = 'C:/Users/abhis/Desktop/Final Year Project/data/glove.6B.300d.txt'
tmp_file = 'C:/Users/abhis/Desktop/Final Year Project/data/word2vec-glove.6B.300d.txt'

from gensim.scripts.glove2word2vec import glove2word2vec
#glove2word2vec(glove_file, tmp_file)
#KeyedVectors.load_word2vec_format(tmp_file).save('vectors.kv')
model = KeyedVectors.load('C:/Users/abhis/Desktop/Final Year Project/data/vectors.kv',mmap='r')





def generate_distractors(answer, count):
    answer = str.lower(answer)
    
    ##Extracting closest words for the answer. 
    try:
        closestWords = model.most_similar(positive=[answer], topn=count)
    except:
        #In case the word is not in the vocabulary, or other problem not loading embeddings
        return []

    #Return count many distractors
    distractors = list(map(lambda x: x[0], closestWords))[0:count]
    
    return distractors


def addDistractors(qaPairs, count):
    for qaPair in qaPairs:
        distractors = generate_distractors(qaPair['answer'], count)
        qaPair['distractors'] = distractors
    
    return qaPairs


# # Main function



def generateQuestions(filename, count):
    text = filename
    
    # Extract words 
    df = generateDf(text)
    wordsDf = prepareDf(df)
    
    # Predict 
    labeledAnswers = predictWords(wordsDf, df)
    
    # Transform questions
    qaPairs = addQuestions(labeledAnswers, text)
    
    # Pick the best questions
    orderedQaPairs = sortAnswers(qaPairs)
    
    # Generate distractors
    questions = addDistractors(orderedQaPairs[:count], 4)

    return questions

import nltk
import nltk.data
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import spacy
import pandas as pd

import nltk
nltk.download('punkt', quiet=True)
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True )
import nltk
nltk.download('wordnet', quiet=True)

nlp = spacy.load('en_core_web_sm')
stemmer = LancasterStemmer()

# List to hold all input sentences
sentences = []

# Dictionary to hold sentences corresponding to respective discourse markers
disc_sentences = {}

# Remaining sentences which do not have discourse markers (To be used later to generate other kinds of questions)
nondisc_sentences = []

# List of auxiliary verbs
aux_list = ['am', 'are', 'is', 'was', 'were', 'can', 'could', 'does', 'do', 'did', 'has', 'had', 'may', 'might', 'must',
            'need', 'ought', 'shall', 'should', 'will', 'would']

# List of all discourse markers
discourse_markers = ['because', 'as a result', 'since', 'when', 'although', 'for example', 'for instance']

# Different question types possible for each discourse marker
qtype = {'because': ['Why'], 'since': ['When', 'Why'], 'when': ['When'], 'although': ['Yes/No'], 'as a result': ['Why'], 
        'for example': ['Give an example where'], 'for instance': ['Give an instance where'], 'to': ['Why']}

# The argument which forms a question
target_arg = {'because': 1, 'since': 1, 'when': 1, 'although': 1, 'as a result': 2, 'for example': 1, 'for instance': 1, 
              'to': 1}

def discourse():
    temp = []
    target = ""
    questions = []
    global disc_sentences
    disc_sentences = {}
    for i in range(len(sentences)):
        maxLen = 9999999
        val = -1
        for j in discourse_markers:
            tmp = len(sentences[i].split(j)[0].split(' '))  
            
            # To get valid, first discourse marker.   
            if(len(sentences[i].split(j)) > 1 and tmp >= 3 and tmp < maxLen):
                maxLen = tmp
                val = j
                
        if(val != -1):

            # To initialize a list for every new key
            if(disc_sentences.get(val, 'empty') == 'empty'):
                disc_sentences[val] = []
                
            disc_sentences[val].append(sentences[i])
            temp.append(sentences[i])


    nondisc_sentences = list(set(sentences) - set(temp))
    
    t = []
    for k, v in disc_sentences.items():
        for val in range(len(v)):
            
            # Split the sentence on discourse marker and identify the question part
            question_part = disc_sentences[k][val].split(k)[target_arg[k] - 1]
            q = generate_question(question_part, qtype[k][0])
            if(q != ""):
                questions.append([disc_sentences[k][val],q])
                
                
    for question_part in nondisc_sentences:
        s = "non_disc"
        sentence = question_part
        text = nltk.word_tokenize(question_part)
        if(text[0] == 'Yes'):
            question_part = question_part[5:]
            s = "Yes/No"
            
        elif(text[0] == 'No'):
            question_part = question_part[4:]
            s = "Yes/No"
            
        q = generate_question(question_part, s)
        if(q != ""):
            questions.append([sentence,q])
        l = generate_one_word_questions(question_part)
        questions += [[sentence,i] for i in l]

    return questions

def sentensify(filez):
    global sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(filez)
    QuE = discourse()
    return QuE

def generate_question(question_part, type):

    ''' 
        question_part -> Part of input sentence which forms a question
        type-> The type of question (why, where, etc)
    '''
    # Remove full stop and make first letter lower case
    question_part = question_part[0].lower() + question_part[1:]
    if(question_part[-1] == '.' or question_part[-1] == ','):
        question_part = question_part[:-1]
        
    # Capitalizing 'i' since 'I' is recognized by parsers appropriately    
    for i in range(0, len(question_part)):
        if(question_part[i] == 'i'):
            if((i == 0 and question_part[i+1] == ' ') or (question_part[i-1] == ' ' and question_part[i+1] == ' ')):
                question_part = question_part[:i] + 'I' + question_part[i + 1: ]
                
    question = ""
    if(type == 'Give an example where' or type == 'Give an instance where'):
        question = type + " " + question_part + '?'
        return question

    aux_verb = False
    res = None
    
    # Find out if auxiliary verb already exists
    for i in range(len(aux_list)):
        if(aux_list[i] in question_part.split()):
            aux_verb = True
            pos = i
            break

    # If auxiliary verb exists
    if(aux_verb):
        
        # Tokeninze the part of the sentence from which the question has to be made
        text = nltk.word_tokenize(question_part)
        tags = nltk.pos_tag(text)
        question_part = ""
        fP = False
        
        for word, tag in tags:
            if(word in ['I', 'We', 'we']):
                question_part += 'you' + " "
                fP = True
                continue
            question_part += word + " "

        # Split across the auxiliary verb and prepend it at the start of question part
        question = question_part.split(" " + aux_list[pos])
        if(fP):
             question = ["were "] + question
        else:
            question = [aux_list[pos] + " "] + question

        # If Yes/No, no need to introduce question phrase
        if(type == 'Yes/No'):
            question += ['?']
            
        elif(type != "non_disc"):
            question = [type + " "] + question + ["?"]
            
        else:
            question = question + ["?"]
         
        question = ''.join(question)

    # If auxilary verb does ot exist, it can only be some form of verb 'do'
    else:
        aux = None
        text = nltk.word_tokenize(question_part)
        tags = nltk.pos_tag(text)
        comb = ""

        '''There can be following combinations of nouns and verbs:
            NN/NNP and VBZ  -> Does
            NNS/NNPS(plural) and VBP -> Do
            NN/NNP and VBN -> Did
            NNS/NNPS(plural) and VBN -> Did
        '''
        
        for tag in tags:
            if(comb == ""):
                if(tag[1] == 'NN' or tag[1] == 'NNP'):
                    comb = 'NN'

                elif(tag[1] == 'NNS' or tag[1] == 'NNPS'):
                    comb = 'NNS'

                elif(tag[1] == 'PRP'):
                    if tag[0] in ['He','She','It']:
                        comb = 'PRPS'
                    else:
                        comb = 'PRPP'
                        tmp = question_part.split(" ")
                        tmp = tmp[1: ]
                        if(tag[0] in ['I', 'we', 'We']):
                            question_part = 'you ' + ' '.join(tmp)
                            
            if(res == None):
                res = re.match(r"VB*", tag[1])
                if(res):
                    
                    # Stem the verb
                    question_part = question_part.replace(tag[0], stemmer.stem(tag[0]))
                res = re.match(r"VBN", tag[1])
                res = re.match(r"VBD", tag[1])

        if(comb == 'NN'):
            aux = 'does'
            
        elif(comb == 'NNS'):
            aux = 'do'
            
        elif(comb == 'PRPS'):
            aux = 'does'
            
        elif(comb == 'PRPP'):
            aux = 'do'
            
        if(res and res.group() in ['VBD', 'VBN']):
            aux = 'did'

        if(aux):
            if(type == "non_disc" or type == "Yes/No"):
                question = aux + " " + question_part + "?"

            else:
                question = type + " " + aux + " " + question_part + "?"
    if(question != ""):
        question = question[0].upper() + question[1:]
    return question

def get_wh_word(entity, sent):
    wh_word = ""
    if entity[1] in ['TIME', 'DATE']:
        wh_word = 'When'
        
    elif entity[1] == ['PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
        wh_word = 'What'
        
    elif entity[1] in ['PERSON']:
            wh_word = 'Who'
            
    elif entity[1] in ['NORP', 'FAC' ,'ORG', 'GPE', 'LOC']:
        index = sent.find(entity[0])
        if index == 0:
            wh_word = "Who"
            
        else:
            wh_word = "Where"
            
    else:
        wh_word = "Where"
    return wh_word

def get_named_entities(sent):
    doc = nlp(sent)
    named_entities = [(X.text, X.label_) for X in doc.ents]
    return named_entities

def generate_one_word_questions(sent):
    
    named_entities = get_named_entities(sent)
    questions = []
    
    if not named_entities:
        return questions
    
    for entity in named_entities:
        wh_word = get_wh_word(entity, sent)
        
        if(sent[-1] == '.'):
            sent = sent[:-1]
        
        if sent.find(entity[0]) == 0:
            questions.append(sent.replace(entity[0],wh_word) + '?')
            continue
       
        question = ""
        aux_verb = False
        res = None

        for i in range(len(aux_list)):
            if(aux_list[i] in sent.split()):
                aux_verb = True
                pos = i
                break
            
        if not aux_verb:
            pos = 9
        
        text = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(text)
        question_part = ""
        
        if wh_word == 'When':
            word_list = sent.split(entity[0])[0].split()
            if word_list[-1] in ['in', 'at', 'on']:
                question_part = " ".join(word_list[:-1])
            else:
                question_part = " ".join(word_list)
            
            qp_text = nltk.word_tokenize(question_part)
            qp_tags = nltk.pos_tag(qp_text)
            
            question_part = ""
            
            for i, grp in enumerate(qp_tags):
                word = grp[0]
                tag = grp[1]
                if(re.match("VB*", tag) and word not in aux_list):
                    question_part += WordNetLemmatizer().lemmatize(word,'v') + " "
                else:
                    question_part += word + " "
                
            if question_part[-1] == ' ':
                question_part = question_part[:-1]
        
        else:
            for i, grp in enumerate(tags):
                
                #Break the sentence after the first non-auxiliary verb
                word = grp[0]
                tag = grp[1]

                if(re.match("VB*", tag) and word not in aux_list):
                    question_part += word

                    if i<len(tags) and 'NN' not in tags[i+1][1] and wh_word != 'When':
                        question_part += " "+ tags[i+1][0]

                    break
                question_part += word + " "
        question = question_part.split(" "+ aux_list[pos])
        question = [aux_list[pos] + " "] + question
        question = [wh_word+ " "] + question + ["?"]
        question = ''.join(question)
        questions.append(question)
    
    return questions

@app.route('/', methods = ['GET','POST'])
def home7():
    return render_template('index2.html')

@app.route('/index.html', methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        message = request.form['message']
        message3 = request.form['message2']
        message2 = int(message3)
        wh = sentensify(message)
        questions = generateQuestions(message, message2)
        q_var = []
        a_var = []

        for z in range(message2):
            q_var.append(questions[z]['question'])
            a_var.append(questions[z]['answer'])
        lst = []
        for i in range(0,message2):
            lst.append(questions[i]['answer'])
        import random

        lstm = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        def top_bottom(word_array,matrix):
            length = len(word_array)
            for i in range(0,340):
                count = 0
                a = random.randint(0,(14-length))
                b = random.randint(0,14)
                for x in range(0,length):
                    if matrix[(a+x)][b] == 0 or matrix[(a+x)][b] == word_array[x]:
                        count = count + 1
                if count == length:
                    for j in range(0,length):
                        matrix[(a+j)][b] = word_array[j]
                    return          

        def left_right(word_array,matrix):
            length = len(word_array)
            for i in range(0,340):
                count = 0
                a = random.randint(0,14)
                b = random.randint(0,(14-length))
                for x in range(0,length):
                    if matrix[a][(b+x)] == 0 or matrix[a][(b+x)] == word_array[x]:
                        count = count + 1
                if count == length:
                    for j in range(0,length):
                        matrix[a][(b+j)] = word_array[j]
                    return          

        def diagonal(word_array,matrix):
            length = len(word_array)
            for i in range(0,340):
                count = 0
                a = random.randint(0,(14-length))
                b = random.randint(0,(14-length))
                for x in range(0,length):
                    if matrix[(a+x)][(b+x)] == 0 or matrix[(a+x)][(b+x)] == word_array[x]:
                        count = count + 1
                if count == length:
                    for j in range(0,length):
                        matrix[(a+j)][(b+j)] = word_array[j]
                    return          

        def fillgaps(matrix,matrix999):
            alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
            alphabets2 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
            for i in range(0,15):
                for j in range(0,15):
                    if matrix[i][j] == 0:
                        n = random.randint(0,25)
                        matrix[i][j] = alphabets[n]
                        matrix999[i][j] = alphabets2[n]


        def choose_fn(choice,array,matrix):
            if choice == 0:
                top_bottom(array,matrix)
                return
            if choice == 1:
                left_right(array,matrix)
                return
            if choice == 2:
                diagonal(array,matrix)
                return

        lst22 = [x.upper() for x in lst]
        for i in range (0,len(lst22)):
            lst222 = list(lst22[i])
            m = random.randint(0,2)
            choose_fn(m,lst222,lstm)
        lstm100 = lstm
        fillgaps(lstm,lstm100)

        matrix1 = []
        for i in range(0,14):
            matrix1.append(lstm[i][0])
        matrix2 = []
        for i in range(0,14):
            matrix2.append(lstm[i][1])
        matrix3 = []
        for i in range(0,14):
            matrix3.append(lstm[i][2])
        matrix4 = []
        for i in range(0,14):
            matrix4.append(lstm[i][3])
        matrix5 = []
        for i in range(0,14):
            matrix5.append(lstm[i][4])
        matrix6 = []
        for i in range(0,14):
            matrix6.append(lstm[i][5])
        matrix7 = []
        for i in range(0,14):
            matrix7.append(lstm[i][6])
        matrix8 = []
        for i in range(0,14):
            matrix8.append(lstm[i][7])
        matrix9 = []
        for i in range(0,14):
            matrix9.append(lstm[i][8])
        matrix10 = []
        for i in range(0,14):
            matrix10.append(lstm[i][9])
        matrix11 = []
        for i in range(0,14):
            matrix11.append(lstm[i][10])
        matrix12 = []
        for i in range(0,14):
            matrix12.append(lstm[i][11])
        matrix13 = []
        for i in range(0,14):
            matrix13.append(lstm[i][12])
        matrix14 = []
        for i in range(0,14):
            matrix14.append(lstm[i][13])
        matrix15 = []
        for i in range(0,14):
            matrix15.append(lstm[i][14])

        mmatrix1 = []
        for i in range(0,14):
            mmatrix1.append(lstm100[i][0])
        mmatrix2 = []
        for i in range(0,14):
            mmatrix2.append(lstm100[i][1])
        mmatrix3 = []
        for i in range(0,14):
            mmatrix3.append(lstm100[i][2])
        mmatrix4 = []
        for i in range(0,14):
            mmatrix4.append(lstm100[i][3])
        mmatrix5 = []
        for i in range(0,14):
            mmatrix5.append(lstm100[i][4])
        mmatrix6 = []
        for i in range(0,14):
            mmatrix6.append(lstm100[i][5])
        mmatrix7 = []
        for i in range(0,14):
            mmatrix7.append(lstm100[i][6])
        mmatrix8 = []
        for i in range(0,14):
            mmatrix8.append(lstm100[i][7])
        mmatrix9 = []
        for i in range(0,14):
            mmatrix9.append(lstm100[i][8])
        mmatrix10 = []
        for i in range(0,14):
            mmatrix10.append(lstm100[i][9])
        mmatrix11 = []
        for i in range(0,14):
            mmatrix11.append(lstm100[i][10])
        mmatrix12 = []
        for i in range(0,14):
            mmatrix12.append(lstm100[i][11])
        mmatrix13 = []
        for i in range(0,14):
            mmatrix13.append(lstm100[i][12])
        mmatrix14 = []
        for i in range(0,14):
            mmatrix14.append(lstm100[i][13])
        mmatrix15 = []
        for i in range(0,14):
            mmatrix15.append(lstm100[i][14])
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        T5_PATH = 't5-large' # T5 model name
        # initialize the model architecture and weights
        t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
        # initialize the model tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
        inputs = t5_tokenizer.encode("summarize:" + message, return_tensors= "pt", max_length=512, padding= 'max_length', truncation=True)
        summary_ids = t5_model.generate(inputs,

                                            num_beams=int(2),

                                            no_repeat_ngram_size=3,

                                            length_penalty=2.0,

                                            min_length= 50,

                                            max_length=200,

                                            early_stopping=True)

        output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output = output.capitalize()
    return render_template('index.html', summary2=output, ques=q_var, answ = a_var, countt = message2, quest = questions, questt = wh, matrixx1 = matrix1, matrixx2 = matrix2, matrixx3 = matrix3, matrixx4=matrix4, matrixx5=matrix5, matrixx6=matrix6, matrixx7=matrix7, matrixx8=matrix8, matrixx9=matrix9, matrixx10=matrix10, matrixx11=matrix11, matrixx12 = matrix12, matrixx13=matrix13, matrixx14=matrix14, matrixx15=matrix15, lstmm = lstm, mmatrixx1 = mmatrix1, mmatrixx2 = mmatrix2, mmatrixx3=mmatrix3, mmatrixx4=mmatrix4, mmatrixx5=mmatrix5, mmatrixx6=mmatrix6, mmatrixx7=mmatrix7, mmatrixx8 = mmatrix8, mmatrixx9=mmatrix9, mmatrixx10=mmatrix10, mmatrixx11=mmatrix11, mmatrixx12=mmatrix12, mmatrixx13=mmatrix13, mmatrixx14=mmatrix14, mmatrixx15=mmatrix15 )

if __name__ == "__main__":
    app.run(debug=True)

