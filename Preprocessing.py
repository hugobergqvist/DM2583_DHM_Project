import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# from textblob import TextBlob
count = 0

def cleaner(datafile):
    print("Preprocessing data...")
    # Global variables
    STOPWORDS = stopwords.words('english')
    cnt = Counter()
    wnl = WordNetLemmatizer()

    for text in datafile:
        for word in text.split():
            cnt[word] += 1

    freq = set([w for (w, wc) in cnt.most_common(10)])

    def removeURLs(sentence):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', sentence)

    def removeEmoji(sentence):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', sentence)

    def increaseCount():
        global count
        count += 1

    """
    def spellCorrection(sentence):
        sentence = TextBlob(sentence)
        sentence = sentence.correct()
        print(count, " sentence is spell corrected")
        increaseCount()
        return sentence
    """

    def lemmatizer(sentence):
        tokens = word_tokenize(sentence)
        token_list = []
        for token in tokens:
            cleaned_token = wnl.lemmatize(token)
            token_list.append(cleaned_token)

        new_sentence = " ".join(token_list)
        return new_sentence

    def removeMostFreqWords(sentence):
        return " ".join([word for word in str(sentence).split() if word not in freq])

    def removeStopwords(sentence):
        return " ".join([word for word in str(sentence).split() if word not in STOPWORDS])

    def punctuations(sentence):
        return re.sub('[^\w\s]', '', sentence)

    def toLower(datafile):
        return datafile.lower()

    def removeHtml(sentence):
        return re.sub('<.*?>', '', sentence)

    #i = 512
    #print(datafile.text[i])

    datafile.text = datafile.text.apply(removeURLs)
    #print("Remove URL done")
    datafile.text = datafile.text.apply(removeHtml)
    #print('Remove Html done')
    datafile.text = datafile.text.apply(toLower)
    #print("ToLower done")
    #datafile.text = datafile.text.apply(spellCorrection) # TRY TO REMOVE AND CHECK ACCURACY
    #print("Spellcheck done")
    datafile.text = datafile.text.apply(punctuations)
    #print("Punctuations done")
    datafile.text = datafile.text.apply(removeStopwords)
    #print("Stopwords done")
    datafile.text = datafile.text.apply(removeMostFreqWords)
    #print("freq words done")
    datafile.text = datafile.text.apply(removeEmoji)
    #print("Remove emojis done")
    #print(datafile.text[i])
    #print(datafile.text[0], " | ", len(datafile.text))
    datafile.text = datafile.text.apply(lemmatizer)

    return datafile
