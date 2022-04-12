import re
from sklearn.feature_extraction.text import TfidfVectorizer

class EMOJI_HANDLING_TFIFDVectorizer(TfidfVectorizer):
    def load_emojis(self):
        filepath = 'd:/batches/SE_PBL/emoji_set.txt'
        fh = open(filepath, 'rb')
        emojis = {}
        for x in fh:
            x = x.decode()
            x = x.strip()
            temp = x.split(' ')
            emojis[temp[0]] = temp[1]

        fh.close()
        return emojis

    #build_analyzer is called on init
    def build_analyzer(self):
        #build_analyzer() of TfidfVectorizer load a method in memory and return a reference to it
        #That method acts on a sentence and breaks it down into words
        #Further those words would have a chance to get added into the vocabulary

        main_analyzer = TfidfVectorizer.build_analyzer(self)
        emojis = self.load_emojis()
        def custom_analyzer(sentence):
            new_sentence = ''
            for x in sentence:
                if x in emojis:
                    new_sentence = new_sentence +  emojis[x]
                else:
                    new_sentence = new_sentence + x

            return main_analyzer(new_sentence)

        return custom_analyzer

class SPAM_HAM_CLASSIFIER:
    def __init__(self, corpus):
        #load the corpus
        fh = open(corpus, 'rb')
        #pre compile a regexp pattern for efficient execution
        pattern = re.compile('(.+)\t(.+)\n')

        self.labels = []
        self.messages = []
        #read the file content line by line
        for x in fh:
            x = x.decode()
            #separate the label and the message
            match_obj = re.search(pattern, x)
            self.labels.append(match_obj.group(1))
            self.messages.append(match_obj.group(2))
        fh.close()

    def create_vocabulary(self):
        #create a vectorizer
        self.punct_tfidf = EMOJI_HANDLING_TFIFDVectorizer(stop_words='english')
        #learn a vocabulary from the messages
        self.punct_tfidf.fit(self.messages)
        print(self.punct_tfidf.get_feature_names())
        #bow
        bow = self.punct_tfidf.transform(self.messages)
        print(bow)


def main():
    try:
        sh_classifier = SPAM_HAM_CLASSIFIER('D:/batches/SE_PBL/SMSSpamCollection')
        sh_classifier.create_vocabulary()
        #... to be continued
    except:
        print('Error')

main()
