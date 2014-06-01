# -*-coding: utf-8-*-
def initialize():
	with open("Stop-Words.txt") as f:
		data = f.read()
	stopWords = data.split("\n")
	for i in "abcçdefgğhıijklmnoöprsştuüvyz":
		if i in stopWords:
			stopWords.remove(i)
	return stopWords
