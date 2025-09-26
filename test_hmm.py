from hmm import get_pos_counts, build_pos_index_maps, get_pi_vector
import nltk
'''
print(nltk.corpus.brown.tagged_sents(tagset='universal')[1][0][1]) #[sentence][word][observation/POS]

#test function that returns the count of each part-of-speech type in a corpous
print(get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]))

posCountsDict = get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])[0]
print(build_pos_index_maps(posCountsDict)[0]) #should print tags
print(build_pos_index_maps(posCountsDict)[1]) #should print pos to index
print(build_pos_index_maps(posCountsDict)[2]) #should print index to pos


#test get_pi_vector function with 
posCountsDict = get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])[0]
totalTokens = get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])[1]
print(get_pi_vector(posCountsDict,totalTokens))


#code to build pi vector
sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
firstWordsOfSentences = [[s[0]] for s in sentences if s] #I put s[0] inside brackets so my get_pos_counts function works still
print(get_pos_counts(firstWordsOfSentences))
posCountsDictStarts = get_pos_counts(firstWordsOfSentences)[0]
totalStartTokens = get_pos_counts(firstWordsOfSentences)[1]
print(get_pi_vector(posCountsDictStarts, totalStartTokens))

'''
print(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])
