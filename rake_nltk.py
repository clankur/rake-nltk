# Author: Ankur Mishra
# Email: 2018amishra@gmail.com
# RAKE Algorithm implemented in Python3 and uses NLTK
# Singularizes nouns, uses cosine similarity to check if phrases are alike, and replaces acronyms
# Specifically designed for parsing medical journals
import operator
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import string
from string import digits
import re
import inflect
import cosine_sim as cs

p = inflect.engine()

sample_text = """
The use of contralateral prophylactic mastectomies (CPMs) among patients with invasive unilateral breast cancer has increased substantially during the past decade in the United States despite the lack of evidence for survival benefit. However, whether this trend varies by state or whether it is correlated with changes in proportions of reconstructive surgery among these patients is unclear.To determine state variation in the temporal trend and in the proportion of CPMs among women with early-stage unilateral breast cancer treated with surgery.A retrospective cohort study of 1.2 million women 20 years of age or older diagnosed with invasive unilateral early-stage breast cancer and treated with surgery from January 1, 2004, through December 31, 2012, in 45 states and the District of Columbia as compiled by the North American Association of Central Cancer Registries. Data analysis was performed from August 1, 2015, to August 31, 2016.Contralateral prophylactic mastectomy.Temporal changes in the proportion of CPMs among women with early-stage unilateral breast cancer treated with surgery by age and state, overall and in relation to changes in the proportions of those who underwent reconstructive surgery.Among the 1 224 947 women with early-stage breast cancer treated with surgery, the proportion who underwent a CPM nationally increased between 2004 and 2012 from 3.6% (4013 of 113 001) to 10.4% (12 890 of 124 231) for those 45 years or older and from 10.5% (1879 of 17 862) to 33.3% (5237 of 15 745) for those aged 20 to 44 years. The increase was evident in all states, although the magnitude of the increase varied substantially across states. For example, among women 20 to 44 years of age, the proportion who underwent a CPM from 2004-2006 to 2010-2012 increased from 14.9% (317 of 2121) to 24.8% (436 of 1755) (prevalence ratio [PR], 1.66; 95% CI, 1.46-1.89) in New Jersey compared with an increase from 9.8% (162 of 1657) to 32.2% (495 of 1538) (PR, 3.29; 95% CI, 2.80-3.88) in Virginia. In this age group, CPM proportions for the period from 2010 to 2012 were over 42% in the contiguous states of Nebraska, Missouri, Colorado, Iowa, and South Dakota. From 2004 to 2012, the proportion of reconstructive surgical procedures among women aged 20 to 44 years who were diagnosed with early-stage breast cancer and received a CPM increased in many states; however, it did not correlate with the proportion of women who received a CPM.The increase in the proportion of CPMs among women with early-stage unilateral breast cancer treated with surgery varied substantially across states. Notably, in 5 contiguous Midwest states, nearly half of young women with invasive early-stage breast cancer underwent a CPM from 2010 to 2012. Future studies should examine the reasons for the geographic variation and increasing trend in the use of CPMs.
"""

l = []
read = open("stat_words.txt", "r") # Statistic words aren't that important in Medical Articles ∴ filter them
for i in read:
    l.append(i.rstrip('\n'))

filter_phrases = set(l)

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text) # POS Tagging

def switchAccs(para): # acronyms are used a lot in medical articles, so we have to switch them out
    para = re.sub(r'\.([a-zA-Z])', r'. \1', para)
    st = para.split(" ")
    acc_to_full_word = dict()
    for i in range(len(st)):
        # st[i] = singularize(st[i])
        if(st[i].startswith("(") and st[i].endswith(")")):
             firstChar = st[i][st[i].index("(")+1]
             acronym = st[i][st[i].index("(")+1:st[i].index(")")]
             acronym = singularize(acronym)
             if(acronym != None):
                 for j in range(i):
                     word = st[i-j].lower()
                     if(len(acronym) > 0 and word.startswith(acronym[0].lower())):
                         full_word = st[i-j:i]
                         acc_to_full_word[acronym] = full_word
    if(len(acc_to_full_word) > 0):
        for i in range(len(st)):
            if(i < len(st) and acc_to_full_word.get(st[i].replace(".", "")) != None):
                st[i:i+1] = acc_to_full_word.get(singularize(st[i].replace(".", "")))
    return " ".join(st)

def is_digit(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False

def singularize(word):
    if(word == "is" or word == "was" or word.endswith("ous") or word.endswith("sis") or word.endswith("xis") or word.endswith("ess") or word == ("thus") or word == ("this") or word.endswith("oss") or word.endswith("ass")):
        return word
    if(p.singular_noun(word) != False):
        word = p.singular_noun(word)
    return word

class RAKE:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.top_fraction = 1 # consider top third candidate keywords by score
    def _generate_candidate_keywords(self, sentences, lower, upper):
        phrase_list = []
        for sentence in sentences:
            words = ["|" if x.lower() in self.stopwords else x for x in nltk.word_tokenize(sentence)]
            for w in range(len(words)):
                words[w] = singularize(words[w])
                words[w] = re.sub('[^A-Za-z|\d\s]+', ' ', words[w])
                words[w] = re.sub(r'_u\d_v\d', '_u%d_v%d', words[w])
            phrase = []
            for word in words:
                if word == "|" or nltk.tag.pos_tag([word])[0][1] == ('IN') or word.endswith("ing") :
                    if len(phrase) >= lower and len(phrase) <= upper :
                        if nltk.tag.pos_tag(phrase[-1])[0][1].startswith('R'):
                            phrase.pop()
                        phrase_list.append(phrase)
                        phrase = []
                else:
                    if word != " ":
                        phrase.append(word)
        return phrase_list

    def _calculate_word_weights(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_weight = nltk.FreqDist()
        for phrase in phrase_list:
            weight = len([x for x in phrase if not is_digit(x)]) - 1
            for x in range(len(phrase)):
                phrase = [x for x in phrase if x]
                # mess with weighting here
                if(phrase[x].lower() in filter_phrases or phrase[x].lower() == 'surgery' or phrase[x].lower() == 'surgical'): # filter these
                    weight = -10
                if(len(phrase[x]) > 5): #if words more complex, gets higher weighting
                    weight = weight + 1
            for word in phrase:
                word_freq.update([word])
                word_weight[word] += weight
        for word in list(word_freq.keys()):
            word_weight[word] = word_weight[word] + 1.5 * word_freq[word] # Frequency > Complexity
        word_weights = {}
        for word in list(word_freq.keys()):
            word_weights[word] = word_weight[word] / word_freq[word]
        return word_weights

    def _calculate_phrase_scores(self, phrase_list, word_weights):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                word = singularize(word)
                if(word_weights.get(word) != None):
                    phrase_score += word_weights[word]
            temp =  " ".join(phrase).lower()
            temp = re.sub(' +',' ', temp)
            phrase_scores[temp] = phrase_score
        # Check if two phrases are similar
        for phrase1 in phrase_list:
            for phrase2 in phrase_list:
                s1 = " ".join(phrase1).lower()
                s2 = " ".join(phrase2).lower()
                if s1 == s2:
                    continue
                vector1 = cs.text_to_vector(s1)
                vector2 = cs.text_to_vector(s2)
                cosine = cs.get_cosine(vector1, vector2)
                if(cosine >= .32):
                    if not (phrase_scores.get(s1) == None or phrase_scores.get(s2) == None):
                        if phrase_scores.get(s1) > phrase_scores.get(s2):
                            phrase_scores.pop(s1, None)
                        elif phrase_scores.get(s1) < phrase_scores.get(s2):
                            phrase_scores.pop(s2, None)
        return phrase_scores

    def extract(self, text, lower, upper):
        text = switchAccs(text)
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences, lower, upper)
        word_weights = self._calculate_word_weights(phrase_list)
        phrase_scores = self._calculate_phrase_scores(
            phrase_list, word_weights)
        sorted_phrase_scores = sorted(iter(phrase_scores.items()),
                                      key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]

def test():
    rake = RAKE()
    keywords = rake.extract(sample_text, 1, 3)
    print("Keywords:", keywords)

if __name__ == "__main__":
    test()
