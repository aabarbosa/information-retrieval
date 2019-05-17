
# Inverted Index for Boolean Search

Simple Inverted Index for politic news retrieval having to choose between linked lists or varying arrays for posting.

>We ran this search in an important communication media of the brazilian society.


#### Modeling and design decisions

- [x] Make Jupyter cells independent,
- [x] Use politic news `id` as ident, and its title as filename,
- [x] Include title for retrieval (important),
- [x] For unix systems, use suitable filename,
- [x] Make corpora compatible to the nltk library.

#### The Dataset (extracted from a paid brazilian portal)


```python
import pandas as pd
politicnews = pd.read_csv('./dataset/news__ptbr.csv', 
                   names=['title', 'content','id'], 
                   header=0)
```

##### Visualization of the dataset
```python
from googletrans import Translator as t # limit: 15k

h = politicnews.head(5); vis = pd.DataFrame(); CHAR=50

#Takes up to ~5 seconds for Google Translator response
en=lambda x: t().translate(x, 
                           src="pt", 
                           dest="en").text[:CHAR]

vis["title"] = h["title"].map(en)
vis["content"] = h["content"].map(en)
vis['id'] = h['id']

print(vis) 
```


| title                         | content                            | id   |
|-----------------------------  |------------------------------------|------|
| 2018 Looking Like 2006        | ...                                  | 1004 |
| Social Division May benefit ..   | Brazil defends political scientist | 1032 |
| ...                         | ...                              | ...  |

#### Details and requirements
By having unlimited and measurable alocated size, linked lists was chosen. Raymond Hettinger’s [OrderedSet](https://orderedset.readthedocs.io/en/latest/) is suitable for fast deletion. It is doubly linked and its implemention can be downloaded [here](https://pypi.org/project/orderedset/#files) and installed by:
```terminal
pip3 install orderedset
``` 
> 1. Python lists are arrays exponentially over-allocated, surprisingly.
> 2. Although provided by the NLTK library, this document do not apply "cleansing" of the portuguese. For example, in exclusion of stoping words, synonyms words, and other usage.

This Orderedset offers O(1) for add, remove, and contains. It is coded in C, and its doubly linked nodes [prev, next, value] are lists. Other data structures would benefit specific scenarios.

> 3. Check requirements{.txt|.py}


```python
ls
```

    [0m[01;34mdataset[0m/                                                       requirements.py
    [01;32minverted-index-for-Boolean-search-in-Brazilian-council.ipynb[0m*  requirements.txt
    inverted-index.py



```python
cat requirements{.txt,.py}
```

    # We recommend to install the latest version, 
    nltk
    orderedset
    
    # Otherwise:
    #nltk-3.4.1
    #orderedset-2.0.1
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # If it lacks or bores you, we recommend to install the corpora:
    #nltk.download( 'all-corpora', quiet=False, raise_on_error=True)
    


#### Creating new corpus, suitable for NLTK library

- Code for filename formatting and implementation for the corpus


```python
from string import punctuation as punct
from nltk.corpus import stopwords as stop
from nltk.tokenize import word_tokenize

def create_unix_filename(s, limit, ext):
    name = [lemma for lemma in word_tokenize(s[:limit].lower()) 
            if lemma not in punct and stop.words('english')
           if len(lemma)>3] #removes "'s", "in", "the", etc
    return ('-'.join(name)+ext)

def create_corpus(folder, dataframe):
    for i, row in dataframe.iterrows():
        ident=row['id']; title=row['title']; content=row['content']
        
        limit=60 #character limit for fileids (less than or equal 255 characters for unix)
        article=str(ident)+'_'+create_unix_filename(title,limit,'.txt') #translate title for results below

        corpus=open(folder+'/'+article,'a')
        corpus.write(' [ '+title+' ] '+content)
        corpus.close()
```

- Make folder for storing the corpus


```sh
%%sh
#!/bin/sh

dir=./dataset
mkdir -p $dir/portuguese_council
ls $dir
```

    news__ptbr.csv
    portuguese_council



```python
%%timeit

create_corpus('./dataset/portuguese_council/', politicnews)

```

    10.1 s ± 123 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


- Instantiate corpus reader for NLTK


```python
from nltk.corpus.reader import PlaintextCorpusReader

corpus=PlaintextCorpusReader('./dataset/portuguese_council/', r'.*\.txt')

len(corpus.fileids())
```




    7643



First 15 fileids (troublesome " 's " already removed, and translation altogether):
<pre>├── 1002_aécio-is-the-best-voted-tucano-in-sp.txt
├── 1004_2018-looking-like-2006.txt
├── 1032_social-division-may-benefit-brazil-defends-political-scientist.txt
├── 1046_aécio-will-accompany-verification-at-her-sister-&apos;s-house-in-bh.txt
├── 1064_aécio-says-he-expects-youssef-to-tell-everything-he-knows.txt
├── 1079_aécio-if-elected-first-mission-is-to-unify-the-country.txt
├── 1094_aécio-votes-in-bh-and-talks-about-unifying-the-country.txt
├── 1114_cid-&apos;s-godson-turns-the-game-over-the-favorite.txt
├── 1128_lobby-agenda-is-waiting-for-the-president.txt
├── 1131_the-election-in-which-it-is-urgent-to-have.txt
├── 1133_aecio-reacts-and-also-uses-his-toolbox.txt
├── 1151_aécio-neves-says-he-will-fire-petrobras-board-if-elected.txt
├── 1160_ácio-accuses-pt-of-terrorism-dilma-highlights-pronatec.txt
├── 1186_aécio-and-dilma-arrive-at-the-last-debate-without-trying-cup.txt
├── 1208_agreement-is-initial-act-of-political-reform-says-janot.txt
(...)</pre>

Use this command for cleansing:

```sh
find ./portuguese_council/ -maxdepth 1 -name "*.txt" -print0 | xargs -0 rm
```


#### The inverted index for indexing news (or articles) 


```python
import nltk, collections 
from orderedset import OrderedSet

def inverted_index(corpus):
    i = collections.defaultdict(OrderedSet)
    for filename in corpus.fileids():
        for term in corpus.words(filename):
            i[term.lower()].add(filename)
    return i
```

#### Search for testing the inverted index (only)


```python
# Warning: This do not replace boolean syntactic analyzers.

def search(i, query):
    """
    :param i: inverted index, case insenstive
    :param query: user input, patterns:
       1-term: if found, a term is returned;
       term1 AND term2: all query terms;
       term1 OR term2: any of the query terms.
    """
    t=lambda s: s.lower().strip()   
    if ' OR ' in query:
        term = query.split(' OR ')
        return i[t(term[0])].union(i[t(term[1])]) 
    elif ' AND ' in query:
        term = query.split(' AND ')
        return i[t(term[0])].intersection(i[t(term[1])])
    else:
        return i[t(query)]  #case insensitive  
```


```python
%%timeit

# Generating the indexes for all news
i = inverted_index(corpus)
```

    48.1 s ± 496 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# Running again for having the index to perform the tests below
i = inverted_index(corpus)
```

#### Testing


```python
from orderedset import OrderedSet

# Lowercased for indexing check (see inverted_index above, and search below)
c=i["campina"]
g=i['grande']

len(c.intersection(g)) # Must be 12 occurrences

```




    12



#### Testing (case insensitive)


```python
assert (len(search(i,"Belo AND Horizonte")) ==  len(search(i,"belo AND horizonte"))) # Case-insensitive assertion
```


```python
assert (len(search(i," latin ")) == len(search(i,"Latin")))
```


```python
assert len(search(i, "debate OR presidencial")) == 1772 # Include titles.

assert len(search(i, "debate AND presidencial")) == 201 

assert len(search(i, "presidenciáveis OR corruptos")) == 164

assert len(search(i, "presidenciáveis AND corruptos")) == 0

assert len(search(i, "Belo OR Horizonte")) == 331

assert len(search(i, "Belo AND Horizonte")) == 242
```

#### Occurences of word "Basque" 


```python
search(i, 'Basque OR Vascos')
```




    OrderedSet()




```python
search(i, 'Basques')
```




    OrderedSet()



#### Occurrences of "Latin"


```python
len(search(i, '                 Latin'))
```




    5




```python
len(search(i, 'Latino'))
```




    20



#### Using orderedset in NLTK corpus


```python
%%timeit

# Bonus Exercise: Using orderedset, compute hit list for ((paris AND NOT france) OR lear)
def dsearch(inverted_index):
    files_with_paris = inverted_index['Paris']
    files_with_france = inverted_index['France']
    files_with_lear = inverted_index['Lear']
    return ((files_with_paris.difference(files_with_france)).union(files_with_lear))

inltk = inverted_index(nltk.corpus.gutenberg) # Creates the corpus

print(dsearch(inltk))
```

    OrderedSet()
    OrderedSet()
    OrderedSet()
    OrderedSet()
    OrderedSet()
    OrderedSet()
    OrderedSet()
    OrderedSet()
    3.61 s ± 20.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


###### Case-sensitive result
OrderedSet(['whitman-leaves.txt'])

#### Other occurrences


```python
len(search(i, 'Italiano'))
```




    50




```python
len(search(i, 'falso OR falsa')) 
```




    92




```python
len(search(i, 'Italiano OR Italiana')) 
```




    73




```python
len(search(i, 'polonês OR polish')) 
```




    3




```python
len(search(i, 'insanidade OR sanidade')) 
```




    2




```python
len(search(i, 'Holandês')) #case insenstive ,
```




    6




```python
search(i, 'England')
```




    OrderedSet()




```python
search(i, 'Reino Unido')
```




    OrderedSet()




```python
search(i,'Grã-Bretanha')
```




    OrderedSet()




```python
len(search(i,'Espanha'))
```




    22



#### Missing features and improvements
- [ ] Check PEP8 convention
- [ ] Reduce the number of syscalls by reducing and unifying achieves
- [x] Reduce filenames by removing words less than or equal 3 characters
- [ ] Improve requirements

### Conslusions

Boolean search is simplistic yet powerful approach for retrieval. The case-sensitive model achieved all indexation in ~8s, against ~16s of its oppose, being possible to search for "Englishman" which returned no results, its oppose returned 30 occurences of unrelated subjects.

By having unlimited and measurable alocated size, linked lists was chosen. The ordered set offered constant time complexity, O(1) for add, remove, and contains. Although other data structures would benefit specific scenarios, excellent results was obtained.


#### References and Thanks
- Leandro Balby for have required the exercise and provided the dataset.
- Joe James for have provided an overview of the NLTK library.




 <footer>
  <p>Suggestions, and more. Contact information at <a href="mailto:aabarbosa.cs@gmail.com">
  gmail</a>.</p>
</footer> 

<style>
footer {
  display: block;
}
    </style>

