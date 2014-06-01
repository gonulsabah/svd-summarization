import dictionary_init

dictionary = dictionary_init.dictionary
suffixes = dictionary_init.suffixes

#######################################
#                                     #
#   DEFAULT CONCEPT IMPLEMENTATIONS   #
#                                     #
#######################################


def search(data, token_type, debug = False):
	global dictionary
	global suffixes
	result = None
	error = False
	if token_type == "word":
		if data[0] in dictionary:
			for each in dictionary[data[0]]:
				if data.startswith(each):
					result = each
		else:
			error = True
		if not error and not result:
			error = True
	elif token_type == "suffix":
		for each in suffixes:
			if data.startswith(each):
				result = each
	if error:
		if token_type == "word":
			with open("NOT_FOUND_WORDS.data","a") as log:
				log.write("\n" + data)
		elif token_type == "suffix":
			with open("NOT_FOUND_SUFFIXES.data","a") as log:
				log.write("\n" + data)
	return result


class Word(object):
	def __init__(self, data, dictionary = dictionary):
		super(Word, self).__init__()
		self.root = search(data, "word")
		self.suffixes = []
		try:
			data = data[len(self.root):]
		except:
			return
		try:
			while data != "":
				self.suffixes.append(search(data,"suffix"))
				data = data[len(self.suffixes[-1]):]
		except TypeError:
# Next line just for debug purposes
#			print("A suffix can not be found. Just FYI...")
			return
	def __repr__(self):
		if self.root:
			if self.suffixes:
				return "<'" + self.root + "'+" + str(self.suffixes) + ">"
			else:
				return "<'" + self.root + "'>"
		else:
			return "<#!NOT FOUND!#>"

class Sentence(object):
	def __init__(self, data):
		super(Sentence, self).__init__()
		self.punctuation = data[-1]
		self.words = [Word(each) for each in data[:-1].split()]
	def __repr__(self):
		return "[" + str(self.words) + ", punctuation:'" + self.punctuation + "']]"
