from __future__ import division
import spacy
import os
import numpy
import math
import csv

def get_elements(first,second,dictionary):
	result = {} 
	for element in dictionary:
		if element[0] == first and element[1] == second:
			result[element] = dictionary[element]

	return result

def get_bi_elements(first,dictionary):
	result = {} 
	for element in dictionary:
		if element[0] == first:
			result[element] = dictionary[element]

	return result

def update_tokens(uni_count,tokens):
	new_tokens = []
	for token in tokens:
		if uni_count[token] == 1:
			new_tokens.append("<unk>")
		else:
			new_tokens.append(token)
	return new_tokens

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

def make_trigram_dict(tokens):
	tricount = {}
	triprobs = {}
	total = 0
	for x in xrange(0,len(tokens)):
		if x+2<len(tokens):
			if (tokens[x],tokens[x+1],tokens[x+2]) in tricount:
				tricount[(tokens[x],tokens[x+1],tokens[x+2])] = tricount[(tokens[x],tokens[x+1],tokens[x+2])] + 1
				total += 1
			else:
				tricount[(tokens[x],tokens[x+1],tokens[x+2])] = 1
				total += 1

	return [tricount,total]

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

corpus = [] 
print "load start"
en_nlp = spacy.load('en')
print "load finish"
"""Code to test individual perplexity calculations"""
for fn in os.listdir('data_corrected/classification task/religion/train_docs'):
    print "running through files"
    if not fn.startswith('.'):
    	corpus = corpus + tokenize('data_corrected/classification task/religion/train_docs/' + fn, en_nlp)


# Get unigram counts and probabilities replacing unknown words
uni_dict = make_unigram_dict(corpus)
uni_count = uni_dict[0]

# UPDATE TOKENS 
corpus = update_tokens(uni_count,corpus)
uni_dict = make_unigram_dict(corpus)
uni_count = uni_dict[0]
uni_total = uni_dict[1]
uni_probs = {}
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

# Get trigram counts with <unk> tags
tri_count = make_trigram_dict(corpus)[0]

# Good turing counts for bigrams
count_dict2 = {}
count_dict[0]= N_0
for count in count_dict:
	#count_dict2[count+1] = count_dict[count]
	if count < 5:
		count_dict2[count] = (count+1)*(count_dict[count+1]/count_dict[count])

# Get trigram probabilities
def get_trigram_probs(bi_count,tri_count):
	tri_probs = {}
	for tri in tri_count:
		#print tri
		tri_probs[tri] = tri_count[tri]/bi_count[(tri[0],tri[1])]

	return tri_probs

def make_trigram_sentence(tri_probs,bi_probs):
	sentence = "<s>"
	bisubset = get_bi_elements("<s>",bi_probs)
	bigrams = bisubset.keys()
	probs = bisubset.values()
	l1,l2 = zip(*bigrams)
	sample = numpy.random.choice(l2,p=probs)

	word1 = "<s>"
	word2 = sample
	end = False
	while end == False:
		subset = get_elements(word1,word2,tri_probs)
		trigrams = subset.keys()
		probs = subset.values()
		list1,list2,list3 = zip(*trigrams)
		print sum(probs)
		sample = numpy.random.choice(list3,p=probs)
		if sample == "</s>":
			end = True
			sentence = sentence +" "+ "</s>"
		elif sample != "<s>":
			sentence = sentence + " " + sample
			word1 = word2
			word2 = sample
	return sentence


tri_gram_probs = get_trigram_probs(bi_count,tri_count)
print make_trigram_sentence(tri_gram_probs,bi_probs)
print bi_count
