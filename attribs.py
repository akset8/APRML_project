
import numpy as np 
import pickle 

latent = pickle.load( open( "latent.p", "rb" ) )

#latent = np.random.randn(160000,100) # load latent here 

# need the latent vector representation first then the addition can happen and then the indexing based on the specific attribute 

print latent.shape

attrib_vectors = np.zeros((40,100),dtype='float32')

print attrib_vectors.shape

import pickle 

with open('celeba.pickle', 'rb') as handle:
    celeba = pickle.load(handle) # dict hai ye 

attrs = []

for i in range(40):

	pos = np.zeros((100),dtype='float32')
	pos_num = 0

	neg = np.zeros((100),dtype='float32')
	neg_num = 0

	key = celeba.keys()

	for img in key:

		#print img
		#print len(celeba[img])

		if(celeba[img][i] == 1 ):

			img = img.split('.')
			ind = int(img[0]) - 1 

			pos = pos + latent[ind , :]
			pos_num = pos_num + 1 

		else:

			img = img.split('.')
			ind = int(img[0]) - 1 

			neg = neg + latent[ind , :]
			neg_num = neg_num + 1 


	print pos_num
	print neg_num

	attr = pos/pos_num - neg/neg_num

	print attr.shape 

	attrs.append(attr)

attrs = np.array(attrs)

import pickle

with open('attribute_vectors.p', 'wb') as handle:

    pickle.dump(attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print attrs.shape 

# get a pickle file out of all those as well 

print attrs









