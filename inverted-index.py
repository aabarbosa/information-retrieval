import nltk, collections

def make_inverted_index(corpus):
    inverted_index = collections.defaultdict(set)
    for filename in corpus.fileids():
        for term in corpus.words(filename):
            inverted_index[term].add(filename)
    return inverted_index


# Exercise Compute hit list for ((paris AND NOT france) OR lear)
def search(inverted_index):
    files_with_paris = inverted_index['Paris']
    files_with_france = inverted_index['France']
    files_with_lear = inverted_index['Lear']
    return ((files_with_paris - files_with_france) or files_with_lear)

inverted_index = make_inverted_index(nltk.corpus.gutenberg)
print (search(inverted_index))




