def unigrams(text):
	return


def tokenize(filename):
	file_obj = open(filename,'r')
	file_str = file_obj.read()
	# Split based on spaces
	file_arr = file_str.split()
	start = False
	start_index = 0
	for x in xrange(0,len(file_arr)):
		if file_arr[x] == 'Subject':
			start_index = x + 2
			start = True
		
		if file_arr[start_index] == "Re":
			start_index = x + 4

		if x >= start_index and start == True:
			# do all tokenizing shit

			print file_arr[x]

	#print file_arr


tokenize("data_corrected/classification task/atheism/train_docs/atheism_file14.txt")
