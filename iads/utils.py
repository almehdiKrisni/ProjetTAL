# Fonctions de chargement des données
import codecs
import re
import string
import unicodedata
import copy

# Libraires pour la récupération des stopwords et le stemming
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
cachedStopWords = stopwords.words("french") + (list(fr_stop))

from nltk.stem.snowball import SnowballStemmer
frStemmer = SnowballStemmer(language='french')

# Chargement des données d'apprentissage sur les présidents
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt)) < 5:
            break
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs

# Chargement des données de tests sur les présidents
def load_pres_test(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt)) < 5:
            break
        lab = re.sub(r"<[0-9]*:[0-9]*(.)>*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*()>*","\\1",txt)
        if lab.count('M') > 0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs

# Fonctions de nettoyage en mise en place des données
def transform(text, punc=False, accentMaj=False, nb=False, stopW=False, stem=False) :
    text_transf = copy.copy(text)
    
    for i in range(len(text)) :
        # Affichage de la ligne
        # if (i % 100 == 0) :
            #print(str(i) + " / " + str(len(text)), end='\r')
        
        # Suppression de la ponctuation
        if punc:
            punc = string.punctuation  # recupération de la ponctuation
            punc += '\n\r\t'
            text_transf[i] = text_transf[i].translate(str.maketrans(punc, ' ' * len(punc))) 
            
         # Suppression des accents et des caractères non normalisés 
        if accentMaj:
            text_transf[i] = unicodedata.normalize('NFD', text_transf[i]).encode('ascii', 'ignore').decode("utf-8")
            text_transf[i] = text_transf[i].lower()
        
        # Suppression des nombres
        if nb:
            text_transf[i] = re.sub('[0-9]+', '', text_transf[i]) # Remplacer une séquence de chiffres par rien
            
        # Suppression des stopwords avec une langue précise
        if stopW :
            text_transf[i] = ' '.join([word for word in text_transf[i].split() if word not in cachedStopWords])
            
        # Utilisation des racines
        if stem :
            text_transf[i] = ' '.join([frStemmer.stem(word) for word in text_transf[i].split()])
            
    return text_transf

# Fonction de suppression des n mots partagés les plus communs
def suppN_sharedmostcommon(n, e1, e2) :
    # Suppression des mots
    suppL = []
    for w1 in e1.most_common(n) :
        for w2 in e2.most_common(n) :
            if (w1[0] == w2[0]) :
                suppL.append(w1[0])
                
    for s in suppL :
        del e1[s]
        del e2[s]
        
    return e1, e2
    
# Fonction permettant de réaliser des tests rapidement
def quickTest(X, Y) :
    # Liste de resultats
    resultSVM = []
    resultNB = []
    resultLR = []
    
    for i in [5, 10] :
        for j in [False, True] :
            for m in range(3) :
                if (m == 0) : # SVC
                    svc = LinearSVC(max_iter=1000)
                    kfold = KFold(n_splits=i, shuffle=j)
                    scores_cv = cross_val_score(svc, Xpres, Ypres, cv=kfold)
                    resultSVM.append(1 - statistics.mean(scores_cv))
                    
                if (m == 1) : # NB
                    clf = MultinomialNB()
                    kfold = KFold(n_splits=i, shuffle=j)
                    scores_cv = cross_val_score(clf, Xpres, Ypres, cv=kfold)
                    resultNB.append(1 - statistics.mean(scores_cv))
                    
                if (m == 2) : # LR
                    lin = LogisticRegression(max_iter=1000)
                    kfold = KFold(n_splits=i, shuffle=j)
                    scores_cv = cross_val_score(lin, Xpres, Ypres, cv=kfold)
                    resultLR.append(1 - statistics.mean(scores_cv))
                    
                
    return resultSVM, resultNB, resultLR