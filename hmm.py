import nltk
import numpy as np
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('brown')
nltk.download('universal_tagset')

#print(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])

#Create a dictionary that maps each POS to the number of times it occurs in the corpus, which is a list that contains lists of tuples
#print(nltk.corpus.brown.tagged_sents(tagset='universal')[1][0][1]) #[sentence][word][observation/POS]

#function that returns the count of each part-of-speech type in a corpos and the token number of tokens in corpus

def get_pos_counts(corpus):
    posCounts = {}
    totalTokens = 0
    for s in corpus: #we select 10,000 sentences in our corpus. for each of those 10000 sentences
        for t in s: #for each tuple in those sentences
            if t[1] in posCounts:
                posCounts[t[1]] +=1
            else:
                posCounts[t[1]] =1

            totalTokens +=1
    #posCounts dictionary is made. Now I want to 

    return (posCounts, totalTokens)

#function that creates a mapping between each part of speech and an assigned index
def build_pos_index_maps(posCounts):
    # "." first, then alphabetical
    tags = sorted(posCounts.keys(), key=lambda t: (t != ".", t))
    pos_to_idx = {t: i for i, t in enumerate(tags)}
    idx_to_pos = {i: t for t, i in pos_to_idx.items()}
    return tags, pos_to_idx, idx_to_pos


#function that return inital probabilities of each probabilities state 
#calculated by (number of this tokens in corpus of type of part of speech)/(number of tokens in corpus)

def get_pi_vector(startPosCounts, totalStartTokens, idx_to_pos):
    pi = np.zeros((len(idx_to_pos),), dtype=np.float64) #set a one dimensional array of length = number of total parts of speech in total corpus
    for i in range(len(idx_to_pos)):
        if idx_to_pos[i] in startPosCounts: #if that pos has ever come at beginning of a sentence
            pi[i] = startPosCounts[idx_to_pos[i]]/totalStartTokens
        else:
            pi[i] = 0
    return pi

def get_A_vector()
    


def build_matrices():
    sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    posCounts, totalTokens = get_pos_counts(sentences)
    tags, pos_to_idx, idx_to_pos = build_pos_index_maps(posCounts)

    firstWordsOfSentences = [[s[0]] for s in sentences if s] #I put s[0] inside brackets so my get_pos_counts function works still
    startPosCounts, totalStartTokens = get_pos_counts(firstWordsOfSentences) #note that some pos types might not occur at sentence start, so might be short
    pi = get_pi_vector(startPosCounts,totalStartTokens, idx_to_pos)




