with open("suffixes.dat") as f:
	data = f.read().split()

data = sorted(list(set(data)))
with open("suffixesV2.dat", "w") as f:
	f.write("\n".join(data))