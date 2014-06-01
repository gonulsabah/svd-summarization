###########################################
#                                         #
#   STANDARD DICTIONARY INITIALIZATIONS   #
#                                         #
###########################################

with open("dictionary.dat", encoding="utf8") as dictionary_file:
	raw_dictionary = sorted(dictionary_file.read().split("\n"))
with open("suffixes.dat", encoding="utf8") as suffix_file:
	suffixes = sorted(suffix_file.read().split("\n"))

dictionary = {}
for each in raw_dictionary:
	if each[0] not in dictionary.keys():
		dictionary[each[0]] = []
	dictionary[each[0]].append(each)

for each in dictionary.keys():
	dictionary[each] = sorted(tuple(set(dictionary[each])))