import spacy

def both_not_in(var1,var2,l):
	if var1 not in l or var2 not in l:
		return True
	else:
		return False

def tokenize(filepath):
	en_nlp = spacy.load('en')

	file_obj = open(filepath,'r')
	file_str = file_obj.read()

	en_doc = en_nlp(file_str.decode('utf8'))

	sentences = [sent.string.strip() for sent in en_doc.sents]
	fixed_sents = []
	for x in range(0,len(sentences)):
		if len(sentences) > (x+1) and len(sentences[x+1]) == 3:
			fixed_sents.append(sentences[x] + sentences[x+1])

		else:
			fixed_sents.append(sentences[x])

	fixed_sents
	word_arr = []
	bad_char_list = ["-","=","/","<",">","#","|"]
	for sent in fixed_sents:
		
		words = sent.split()
		#if words[0].encode('utf-8') = "From" and words[0].encode('utf-8') == "Subject":
		#	print words[0].encode('utf-8')
		if words[0].encode('utf-8') != "From" and words[0].encode('utf-8') != "Subject":
			word_arr.append("<s>")
			for x in range(0,len(words)):
				if (x+1) < len(words):
					if words[x] != "." and both_not_in(words[x],words[x+1],bad_char_list):
						word_arr.append(words[x].encode('utf-8'))

			word_arr.append("</s>")		

	return word_arr


def make_bigram_dict(tokens):
	bicount = {}

	for x in xrange(0,len(tokens)):
		if x+1<len(tokens):
			if (tokens[x],tokens[x+1]) in bicount:
				bicount[(tokens[x],tokens[x+1])] = bicount[(tokens[x],tokens[x+1])] + 1
			else:
				bicount[(tokens[x],tokens[x+1])] = 1

	print bicount

make_bigram_dict(tokenize("data_corrected/classification task/medicine/train_docs/medicine_file21.txt"))

