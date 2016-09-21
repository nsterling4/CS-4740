from __future__ import division
import spacy
import os
import numpy
import math


def both_not_in(var1,var2,l):
	if var1 not in l or var2 not in l:
		return True
	else:
		return False

def tokenize(filepath, en_nlp):
	file_obj = open(filepath,'r')
	file_str = file_obj.read()


	en_doc = en_nlp(file_str.decode('utf8'))

	sentences = [sent.string.strip() for sent in en_doc.sents]

	fixed_sents = []
	file_obj.close()
	for x in range(0,len(sentences)):
		if len(sentences) > (x+1) and len(sentences[x+1]) == 3:
			fixed_sents.append(sentences[x] + sentences[x+1])

		else:
			fixed_sents.append(sentences[x])


	word_arr = []
	bad_char_list = ["-","=","/","<",">","#","|","(",")","[","]","*","'",'"',";",":","`",",","@","+"]
	for sent in fixed_sents:
		
		words = sent.split()
		if words[0].encode('utf-8') != "From" and words[0].encode('utf-8') != "Subject" :
			word_arr.append("<s>")
			for x in range(0,len(words)):
				if (x+1) < len(words):
					if words[x] != "." and words[x] != "?" and words[x] != "!" and words[x] not in bad_char_list and words[x].find("@") == -1 and words[x].find(".") == -1:
						word_arr.append(words[x].encode('utf-8').lower())

			word_arr.append("</s>")		

	return word_arr


def make_bigram_dict(tokens):
	bicount = {}
	biprobs = {}
	total = 0
	for x in xrange(0,len(tokens)):
		if x+1<len(tokens):
			if (tokens[x],tokens[x+1]) in bicount:
				bicount[(tokens[x],tokens[x+1])] = bicount[(tokens[x],tokens[x+1])] + 1
				total += 1
			else:
				bicount[(tokens[x],tokens[x+1])] = 1
				total += 1

	return [bicount,total]

def make_unigram_dict(tokens):
	unicount = {}
	uniprobs = {}
	total = 0

	for x in xrange(0,len(tokens)):
		if tokens[x] in unicount:
			unicount[tokens[x]] = unicount[tokens[x]] + 1
			total += 1
		else:
			unicount[tokens[x]] = 1
			total += 1			
	return [unicount,total]

def get_elements(first,dictionary):
	result = {} 
	for element in dictionary:
		if element[0] == first:
			result[element] = dictionary[element]

	return result


def make_unigram_sentence(uni_probs):
	words = uni_probs.keys()
	probs = uni_probs.values()
	end = False
	sentence = "<s>"
	while end == False:
		word = numpy.random.choice(words,p=probs)
		if word == "</s>":
			end = True
			sentence = sentence + " " + word
		elif word != "<s>":
			sentence = sentence + " " + word
			

	return sentence

def make_bigram_sentence(bi_probs):
	sentence = "<s>"
	word = "<s>"
	end = False
	while end == False:
		subset = get_elements(word,bi_probs)
		bigrams = subset.keys()
		probs = subset.values()
		list1,list2 = zip(*bigrams)
		sample = numpy.random.choice(list2,p=probs)
		if sample == "</s>":
			end = True
			sentence = sentence +" "+ "</s>"
		elif sample != "<s>":
			sentence = sentence + " " + sample
			word = sample

	return sentence


def calc_perplexity(bi_probs,test_file_tokens):
	#calculating perplexity from bigram probability

	#Get all bigrams from test_file_tokens
	acc = 0.0
	test_bigrams = []
	min_prob = min(bi_probs, key=bi_probs.get)
	for x in xrange(0,len(test_file_tokens)):
		if x+1<len(test_file_tokens):
			test_bigrams.append((test_file_tokens[x],test_file_tokens[x+1]))

	test_file_probs = {}
	for bigram in test_bigrams:
		if bigram in bi_probs.keys():
			prob = bi_probs[bigram]
		else:
			prob = bi_probs[min_prob]
		
		if prob > 0.0:
			exp = -math.log(prob)
			acc += exp

	result = acc ** (1.0/len(test_file_tokens))

	return result




corpus = [] 
print "load start"
en_nlp = spacy.load('en')
print "load finish"
for fn in os.listdir('data_corrected/classification task/space/train_docs'):
    print "running through files"
    if not fn.startswith('.'):
    	corpus = corpus + tokenize('data_corrected/classification task/space/train_docs/' + fn, en_nlp)
#corpus = tokenize('data_corrected/classification task/medicine/test_medicinefile1.txt',en_nlp)

uni_dict = make_unigram_dict(corpus)
uni_count = uni_dict[0]
uni_total = uni_dict[1]
uni_probs = {}
#getting unigram probabilities
for uni in uni_count:
	uni_probs[uni] = uni_count[uni]/uni_total


bi_dict = make_bigram_dict(corpus)
bi_count = bi_dict[0]
bi_total = bi_dict[1]
bi_probs = {}
count_dict = {}
N_0 = (len(uni_count.keys())**2) - len(bi_count.keys())

# Getting bigram probabilities
for bi in bi_count:
	bi_probs[bi] = bi_count[bi]/uni_count[bi[0]]
	# Get dictionary of all bigrams and counts for smoothing
	# Each entry is composed of a count, N, and how many bigrams occurred with that N
	if bi_count[bi] not in count_dict:
		count_dict[bi_count[bi]] = 1
	else:
		count_dict[bi_count[bi]] = count_dict[bi_count[bi]] + 1

# Adjust unigram and bigram counts to have unknown words counted
adj_uni_count = {}
adj_uni_count["<unk>"] = 1
for uni in uni_count:
	if uni_count[uni] == 1:
		adj_uni_count["<unk>"] = adj_uni_count["<unk>"] + 1
	else:
		adj_uni_count[uni] = uni_count[uni]
#print adj_uni_count

adj_bi_count = {}
for bi in bi_count:
	if bi[0] not in adj_uni_count.keys():
		adj_bi_count[("<unk>",bi[1])] = bi_count[bi]
	elif bi[1] not in adj_uni_count.keys():
		adj_bi_count[(bi[0],"<unk>")] = bi_count[bi]
	else:
		adj_bi_count[bi] = bi_count[bi]

#print adj_bi_count

count_dict2 = {}
count_dict[0]= N_0
for count in count_dict:
	#count_dict2[count+1] = count_dict[count]
	if count < 5:
		count_dict2[count] = (count+1)*(count_dict[count+1]/count_dict[count])

#print count_dict2

#PERPLEXITY
# for fn in os.listdir('data_corrected/classification task/test_for_classification'):
#     print "running through files"
#     	if not fn.startswith('.'):
#     		test_file = tokenize('data_corrected/classification task/test_for_classification/' + fn, en_nlp)
print "tokenizing test_file for PERPLEXITY"
test_file = tokenize('data_corrected/classification task/test_for_classification/file_0.txt', en_nlp)
print "Perplexity =",calc_perplexity(bi_probs,test_file)


# print "\n"
# print "Display Unigram Model Sentences\n"
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print make_unigram_sentence(uni_probs)
# print "\n"
# print "Display Bigram Model Sentences\n"
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)
# print make_bigram_sentence(bi_probs)

