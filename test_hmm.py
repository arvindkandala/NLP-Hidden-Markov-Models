from hmm import build_matrices, get_obs_seq, get_pos_counts, build_pos_index_maps
from viterbi import viterbi
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



pi, A, B = build_matrices(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])
print(B.shape)


'''

trainingCorpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
pi, A, B = build_matrices(trainingCorpus)

posCounts, _ = get_pos_counts(trainingCorpus)
tags, pos_to_idx, idx_to_pos = build_pos_index_maps(posCounts)

'''
test_slice = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
obs = get_obs_seq(test_slice, trainingCorpus) 

qs, prob = viterbi(obs, pi, A, B)

pred_tags = [idx_to_pos[i] for i in qs]
print("probability", prob)
print(pred_tags)

accuracyTable = []
wordCount = 0
trueCount = 0
for sent in test_slice:
    for i in range(len(sent)):
        wordCount+=1
        accuracyTable.append(sent[i][1] == pred_tags[i])
        if(sent[i][1] == pred_tags[i]):
            trueCount+=1
print(accuracyTable)
print("accuracy for putting all 3 sentences together:", trueCount/wordCount) #accuracy was .34
'''

sent10150 = nltk.corpus.brown.tagged_sents(tagset='universal')[10150]
obs2 = get_obs_seq([sent10150], trainingCorpus) #need to put single_sent in bracket for get_obs_seq to work
qs2, prob2 = viterbi(obs2, pi, A, B)
pred_tags2 = [idx_to_pos[i] for i in qs2]
words2 = [w for (w, _) in sent10150]
print("prob:", prob2)
print(pred_tags2)
accuracyTable2 = []
wordCount2 = 0
trueCount2 = 0
for i in range(len(sent10150)):
    wordCount2+=1
    accuracyTable2.append(sent10150[i][1] == pred_tags2[i])
    if(sent10150[i][1] == pred_tags2[i]):
        trueCount2+=1
print(accuracyTable2)
print("accuracy for sentence 10150:", trueCount2/wordCount2) #accuracy of .92
print("\nword\t\tpred_tag")
for w, t in zip(words2, pred_tags2):
    print(f"{w}\t\t{t}")



sent10151 = nltk.corpus.brown.tagged_sents(tagset='universal')[10151]
obs3 = get_obs_seq([sent10151], trainingCorpus) #need to put single_sent in bracket for get_obs_seq to work
qs3, prob3 = viterbi(obs3, pi, A, B)
pred_tags3 = [idx_to_pos[i] for i in qs3]
words3 = [w for (w, _) in sent10151]
print("prob:", prob3)
print(pred_tags3)
accuracyTable3 = []
wordCount3 = 0
trueCount3 = 0
for i in range(len(sent10151)):
    wordCount3+=1
    accuracyTable3.append(sent10151[i][1] == pred_tags3[i])
    if(sent10151[i][1] == pred_tags3[i]):
        trueCount3+=1
print(accuracyTable3)
print("accuracy for sentence 10151:", trueCount3/wordCount3) #accuracy of .89
print("\nword\t\tpred_tag")
for w, t in zip(words3, pred_tags3):
    print(f"{w}\t\t{t}")



sent10152 = nltk.corpus.brown.tagged_sents(tagset='universal')[10152]
obs4 = get_obs_seq([sent10152], trainingCorpus) #need to put single_sent in bracket for get_obs_seq to work
qs4, prob4 = viterbi(obs4, pi, A, B)
pred_tags4 = [idx_to_pos[i] for i in qs4]
print("prob:", prob4)
words4 = [w for (w, _) in sent10152]
accuracyTable4 = []
wordCount4 = 0
trueCount4 = 0
for i in range(len(sent10152)):
    wordCount4+=1
    accuracyTable4.append(sent10152[i][1] == pred_tags4[i])
    if(sent10152[i][1] == pred_tags4[i]):
        trueCount4+=1
print(accuracyTable4)
print("accuracy for sentence 10152:", trueCount4/wordCount4) #accuracy of 1.0
print("\nword\t\tpred_tag")
for w, t in zip(words4, pred_tags4):
    print(f"{w}\t\t{t}")
