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
def get_pi_vector(posCounts, totalTokens):
    pi = np.zeros((len(posCounts),), dtype=np.float64) #set a one dimensional array of length of posCounts
    #get indices of each pos
    mapping = build_pos_index_maps(posCounts)
    idx_to_pos = mapping[2]
    for i in range(len(posCounts)):
        pi[i] = posCounts[idx_to_pos[i]]/totalTokens
    return pi




    




