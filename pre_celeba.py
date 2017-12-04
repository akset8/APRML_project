import os 
import glob 

f = open('list_attr_celeba.txt','r')

li = f.readlines()[:160000]

celeba = {}

for i in li:

	out = i 

	out = out.split(' ')

	attrs = []

	img = out[0]

	for i in range(1,len(out)):

		try:
			attrs.append(int(out[i]))
		
		except:
			pass



	print len(attrs)

	celeba[img] = attrs

import pickle


for img in celeba.keys():

	out = celeba[img]

	for i in range(len(out)):

		if(out[i] == -1):
			out[i] = 0

print celeba

with open('celeba.pickle', 'wb') as handle:
    pickle.dump(celeba, handle, protocol=pickle.HIGHEST_PROTOCOL)


