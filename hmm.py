import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('brown')
nltk.download('universal_tagset')

print(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])

