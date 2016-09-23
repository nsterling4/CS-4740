import csv
r = csv.reader(open("perplexity_final.csv", "r"))
c = csv.writer(open("perplexity_w_class.csv", "wb"),delimiter=',')
next(r, None)  # skip the headers
topics = ["motorcycles","medicine","atheism","autos","graphics","religion","space"]
for row in r:
	nums = [row[1],row[2],row[3],row[4],row[5],row[6],row[7]]
	min_perplex = nums.index(min(nums))
	c.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],topics[min_perplex]])