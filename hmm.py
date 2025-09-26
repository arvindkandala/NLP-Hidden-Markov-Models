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
    denom = totalStartTokens + len(idx_to_pos)
    for i in range(len(idx_to_pos)):
        if idx_to_pos[i] in startPosCounts: #if that pos has ever come at beginning of a sentence
            pi[i] = (startPosCounts[idx_to_pos[i]]+1)/denom
        else:
            pi[i] = 1/denom
    return pi

def get_A_matrix(pos_to_idx, sentences):
    q = len(pos_to_idx)
    A = np.zeros((q, q), dtype=np.float64)

    for posPre in pos_to_idx:
        posPostCounts = {k: 1 for k in pos_to_idx} #start at 1 instead of 0 for add-1 smoothing
        posPreCount = 0 #contains the number of times each pos occurs after our pos prefix
        for sent in sentences: 
            for i in range(len(sent)): #for each word in each sentence
                if sent[i][1] == posPre:
                    posPreCount += 1
                    if i<len(sent)-1:
                        posPostCounts[sent[i+1][1]] += 1 #count     

        for posPost in posPostCounts:
            posPostCounts[posPost] = posPostCounts[posPost]/(posPreCount+q) #now we have the probability of each posPost coming after posPre, q added for smoothing
            A[pos_to_idx[posPre], pos_to_idx[posPost]] = posPostCounts[posPost]
    return A

def get_vocab_with_UNK(sentences):
    vocab = set()
    for sent in sentences:
        for tup in sent:
            vocab.add(tup[0])
    vocab.add('UNK')
    return vocab

def get_B_matrix(pos_to_idx, sentences, posCounts, idx_to_pos):
    vocab = get_vocab_with_UNK(sentences)
    word_to_idx = {w: i for i, w in enumerate(sorted(vocab))}
    q = len(pos_to_idx)
    v = len(vocab)
    B = np.zeros((q, v), dtype=np.float64)

    for sent in sentences:
        for tup in sent:
            i = pos_to_idx[tup[1]]
            if tup[0] in vocab:
                j = word_to_idx[tup[0]]
            else:
                j = word_to_idx['UNK']
            B[i,j] += 1
        
    for i in range(q):
        for j in range(v):
            B[i,j] += 1 #add-1 smoothing

    for i in range(q):
        denom = posCounts[idx_to_pos[i]] + v
        for j in range(v):
            B[i,j] = B[i,j] / denom #part of smoothing
        
    return B    


def build_matrices(corpus):
    sentences = corpus #nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    posCounts, totalTokens = get_pos_counts(sentences)
    tags, pos_to_idx, idx_to_pos = build_pos_index_maps(posCounts)

    firstWordsOfSentences = [[s[0]] for s in sentences if s] #I put s[0] inside brackets so my get_pos_counts function works still
    startPosCounts, totalStartTokens = get_pos_counts(firstWordsOfSentences) #note that some pos types might not occur at sentence start, so might be short
    pi = get_pi_vector(startPosCounts,totalStartTokens, idx_to_pos)

    A = get_A_matrix(pos_to_idx, sentences)

    B = get_B_matrix(pos_to_idx,sentences,posCounts,idx_to_pos)

    return(pi, A, B)

def get_obs_seq(testingCorpus, sentences):
    vocab = get_vocab_with_UNK(sentences)
    word_to_idx = {w: i for i, w in enumerate(sorted(vocab))}
    obs = []
    for sent in testingCorpus:
        for tup in sent:
            obs.append(word_to_idx.get(tup[0], word_to_idx['UNK']))
    return obs