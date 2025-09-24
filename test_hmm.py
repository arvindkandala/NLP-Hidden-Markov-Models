from hmm import get_pos_counts, build_pos_index_maps
import nltk

#print(nltk.corpus.brown.tagged_sents(tagset='universal')[1][0][1]) #[sentence][word][observation/POS]

#test function that returns the count of each part-of-speech type in a corpous
#print(get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]))

posCountsDict = get_pos_counts(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])[0]
print(build_pos_index_maps(posCountsDict)[0]) #should print tags
print(build_pos_index_maps(posCountsDict)[1]) #should print pos to index
print(build_pos_index_maps(posCountsDict)[2]) #should print index to pos
