#-*- coding: utf-8 -*-
import locale
from numpy import zeros
from scipy.linalg import svd
#following needed for TFIDF
from math import log, fabs, ceil
from numpy import asarray, sum
from numpy import matrix
from stopword import initialize
import os
import re

from nlp_concept_definitions import *


os.environ["CLASSPATH"] = ".:" + os.path.dirname(os.path.realpath(__file__)) + "/*"

def string_to_vector(string):
	string = string.replace("'", "")
	string = string.replace('"', "")
	wordlist = re.findall("[A-Za-z0-9\ç\ğ\ı\ö\ş\ü\Ç\Ğ\İ\Ö\Ş\Ü-]+", string)
	dictionary = {}
	for word in wordlist:
		if word in dictionary:
			dictionary[word] += 1
		else:
			dictionary[word] = 1
	return dictionary


def dot_product_text(vector1, vector2):
	intersection = []
	for key in set(list(vector1.keys()) + list(vector2.keys())):
		if key in vector1.keys() and key in vector2.keys():
			intersection.append(key)
	result = 0
	for key in intersection:
		result += vector1[key] * vector2[key]
	return result

def cosine_similarity(sentence1, sentence2):
	sentence1 = string_to_vector(sentence1)
	sentence2 = string_to_vector(sentence2)
	return dot_product_text(sentence1, sentence2) / (dot_product_text(sentence1, sentence1) * dot_product_text(sentence2, sentence2))**(1 / 2)

#sentences = ["The Neatest Little Guide to Stock Market Investing",
#	  "Investing For Dummies, 4th Edition",
#	  "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
#	  "The Little Book of Value Investing",
#	  "Value Investing: From Graham to Buffett and Beyond",
#	  "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
#	  "Investing in Real Estate, 5th Edition",
#	  "Stock Investing For Dummies",
#	  "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
#	  ]
#stopwords = ['and','edition','for','in','little','of','the','to']


lower_tr = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ", "abcçdefgğhıijklmnoöpqrsştuüvwxyz")
upper_tr = str.maketrans("abcçdefgğhıijklmnoöpqrsştuüvwxyz", "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ")

def text_tokenize(text):
	result = []
	c = Sentence(text)
	print(c)
	return " ".join([token.root for token in c.words if token.root])

def document_read(path):
	with open(path, encoding="utf8") as f:
		data = f.read().split("\n")

	for i in range(len(data)):
		data[i] = data[i].translate(lower_tr)
		for punc in [',', ':', "'", '!', '"', "!", "?", "(", ")", "[", "]"]:
			data[i] = data[i].replace(punc, "")	
	return data


try:
	sentences = document_read(input("dosya>"))

except EOFError:
	print("BURADAN DEGIL KONSOLDAN CALISTIR ARTIK!")
	exit()

ignorechars = '''()",:'!'''

stopwords = initialize()

class LSA(object):
	def __init__(self, stopwords, ignorechars):
		self.stopwords = stopwords
		self.ignorechars = ignorechars
		self.wdict = {}
		self.dcount = 0
		self.sentences = []
		self.root_sentences = []

	def parse(self, doc):
		words = re.findall("[A-Za-z0-9\ç\ğ\ı\ö\ş\ü\Ç\Ğ\İ\Ö\Ş\Ü-]+", doc)
		for punc in [',', ':', "'", '!', '"', ".", "!", "?", "(", ")", "[", "]"]:
			doc = doc.replace(punc, "")
		for w in words:
			#w = w.lower().translate("", self.ignorechars)
			w = w.translate(lower_tr)
			if w in self.stopwords:
				continue
			elif w in self.wdict:
				self.wdict[w].append(self.dcount)
			else:
				self.wdict[w] = [self.dcount]
		self.dcount += 1
		self.sentences.append(doc)
		self.root_sentences.append(text_tokenize(doc))


	def build(self):
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		self.A = zeros([len(self.keys), self.dcount])
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				self.A[i,d] += 1



	def calc(self):
		view_matrix = []
		for row in self.A: # EDIT-TAG
			view_matrix.append(list([int(value > 0) for value in row]))
		
		print("View matrix:")
		print(*view_matrix, sep="\n")
		
		Y_section_density = [] # EDIT-TAG

		for row in view_matrix: 
			Y_section_density.append(sum(row))

		print("Section density (per word) : ",Y_section_density)

		first_views = [] # EDIT-TAG

		for row in view_matrix: 
			first_views.append(row.index(1))

		print("First views (per word) : ", first_views)

		last_views = [] # EDIT-TAG

		for row in view_matrix:
			last_views.append(len(row) - 1 - list(reversed(row)).index(1))

		print("Last views (per word) : ", last_views)

		term_counts = [] # EDIT - TAG

		for row in self.A:
			term_counts.append(int(sum(row)))

		print("Term counts (per word) : ", term_counts)

		view_density = list([last_views[i] - first_views[i] for i in range(len(first_views))]) # EDIT-LINE
		print("View density (per word) : ", view_density)


		moments = [] # EDIT - TAG
		for j in range(len(self.A)):
			moment = 0
			for i,ci in list(enumerate(self.A[j])):
				moment += i*ci
			moments.append(moment / term_counts[j])

		print("Moments (per word) : ", moments)

		position_variances = [] # EDIT - TAG

		for j in range(len(self.A)):
			variance = 0
			for i,ci in list(enumerate(self.A[j])):
				variance += ci * fabs(i - moments[j])
			position_variances.append(variance / term_counts[j])

		print("Position variances (per word) : ", position_variances)

		T_distribution = [(Y_section_density[i] + view_density[i] + position_variances[i]) / 3  for i in range(len(self.A))] # EDIT - TAG

		print("Distributional value (per word) : ", T_distribution)

		score_1 = [] # EDIT - TAG
		for i in range(len(self.root_sentences)):
			score_1.append(fabs(len(self.A[0]) - 2 * i) / len(self.A[0]))

		score_2 = []
		for s in self.root_sentences:
			score_2.append(cosine_similarity(s, self.root_sentences[1]))

		score_3 = []
		for s in self.root_sentences:
			score_3.append(cosine_similarity(s, self.root_sentences[-1]))

		score_4 = []
		for s in self.root_sentences:
			score_4.append(cosine_similarity(s, self.root_sentences[0]))

		score_5 = []
		word_count = sum(list(string_to_vector(" ".join(self.root_sentences)).values()))
		for s in self.root_sentences:
			score_5.append(sum(list(string_to_vector(s).values())) / word_count)

		score_6 = []
		words = {}

		for s in self.root_sentences:
			v = string_to_vector(s)
			for key in v.keys():
				if key in words:
					words[key] += v[key]
				else:
					words[key] = v[key]

		for sword in stopwords:
			if sword in words:
				del words[sword]
		 
		frequency_table = sorted(list(words.items()), key = lambda x : x[1], reverse = True)
		print("frequency_table:", frequency_table)
		print("len(frequency_table):", len(frequency_table))
		criteria = frequency_table[:ceil(len(frequency_table) / 10)]
		print("criteria:", criteria)
		print("len(criteria):", len(criteria))

		criteria_value_sum = sum([count for word, count in criteria])
		for s in self.root_sentences:
			score = 0
			for word,count in criteria:
				vector = string_to_vector(s)
				if word in vector:
					score += vector[word]
			score_6.append(score / criteria_value_sum)

		conclusion_words = ["neticede", "sonuçta", "sonuç olarak", "özetle", "toparlarsak", "bitirirken", "nihayet", "son olarak",]

		score_7 = []
		for s in self.sentences:
			score = 0
			for cword in conclusion_words:
				score += s.count(cword)
			score_7.append(score)
		for i in range(len(score_7)):
			score_7[i] /= max(1, sum(score_7))

		print("Score 1:", score_1)
		print("Score 2:", score_2)
		print("Score 3:", score_3)
		print("Score 4:", score_4)
		print("Score 5:", score_5)
		print("Score 6:", score_6)
		print("Score 7:", score_7)

		score_total = [sum((score_1[i], score_2[i], score_3[i], score_4[i], score_5[i], score_6[i], score_7[i])) for i in range(len(score_1))]
		print("total_score:", score_total)

		##################### HERE ########################

		tdistribution_numpy = matrix(T_distribution)
		score_total_numpy = matrix(score_total)
		print("A:",len(self.A.T), "x", len(self.A))
		print("tdistribution_numpy:", len(tdistribution_numpy.T), "x", len(tdistribution_numpy))
		print("score_total:", len(score_total_numpy), "x", len(score_total_numpy.T))
		weight_matrix = self.A.T * tdistribution_numpy.T
		weight_matrix = weight_matrix * score_total_numpy
		print("weight", len(weight_matrix), "x", len(weight_matrix.T))

		self.U, self.S, self.Vt = svd(weight_matrix) #LEAD-LINE

	def TFIDF(self):
		WordsPerDoc = sum(self.A, axis=0)	 
		DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
				self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])

	def printA(self):
		print('Here is the count matrix')
		print(self.A)

	def printSVD(self):
		print('Here are the singular values')
		print(self.S)
		print('Here is the U matrix')
		print(-1*self.U[:,:])
		print('Here is the Vt matrix')
		print(-1*self.Vt[:,:])

print(*sentences, sep="\n")


mylsa = LSA(stopwords, ignorechars)
for t in sentences[1:]:
	mylsa.parse(t)
mylsa.build()
#mylsa.printA()
mylsa.calc()
#mylsa.printSVD()

summary_matrix = [list(row) for row in list(mylsa.Vt)]
print(*summary_matrix, sep = "\n")
print("-----")
for row in summary_matrix:
	for i in range(len(row)):
		if row[i] < sum(row) / len(row):
			row[i] = 0

summary_scores = []
for j in range(len(summary_matrix[0])):
	total = 0
	for i in range(len(summary_matrix)):
		total += summary_matrix[i][j]
	summary_scores.append(total)

print(summary_scores)

summary_indexes = [i for i in range(len(summary_scores)) if summary_scores[i] > sum(summary_scores) / len(summary_scores)]

print(mylsa.sentences)

print("--- summary ---")
for each in summary_indexes:
	print(mylsa.sentences[each])
print("--- summary ---")