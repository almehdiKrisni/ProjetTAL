# Fonctions de chargement des données
import codecs
import re
import string
import unicodedata
from nltk.corpus import stopwords

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
def transform(text, punc=False, accentMaj=False, nb=False, stopW=False, base=stopwords.words('french')) :
    text_transf = text
    
    for i in range(len(text)) :
        # Affichage de la ligne
        if (i % 100 == 0) :
            print(str(i) + " / " + str(len(text)), end='\r')
        
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
            text_transf[i] = ' '.join([word for word in text_transf[i].split() if word not in base])
            
    return text_transf