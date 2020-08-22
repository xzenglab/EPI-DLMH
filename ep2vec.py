# system modules
import os
import time
import sys
# gensim modules
from gensim import utils
from smart_open import smart_open_lib
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# random shuffle
from random import shuffle
# numpy
import numpy 
# classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
enhancers_num = 0
promoters_num = 0
positive_num  = 0
negative_num  = 0

#kmer = int(sys.argv[1]) # the length of k-mer
#swin = int(sys.argv[2]) # the length of stride
#vlen = int(sys.argv[3]) # the dimension of embedding vector
#cl   = sys.argv[4]      # the interested cell line


#bed2sent: convert the enhancers.bed and promoters.bed to kmer sentense
#bed format: chr start end name
def bed2sent(filename,k,win):
	fin   = open(filename,'r')
	fout  = open('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.sent','w')
	for line in fin:
		if line[0] =='>':
			continue
		else:
			line   = line.strip().lower()
			length = len(line)
			i = 0
			while i<= length-k:
				fout.write(line[i:i+k]+' ')
				i = i + win
			fout.write('\n')

def genarare_file():
	enhancer_train = open('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.sent', 'r')
	enhancer_test = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_enhancer_test.sent','r')
	enhancer_all = open('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer_all.sent', 'w')
	for line in enhancer_train:
		enhancer_all.write(line)
		enhancer_all.write('\n')
	for line in enhancer_test:
		enhancer_all.write(line)
		enhancer_all.write('\n')
	enhancer_all.close()

	promoter_train = open('/home/ycm/data/GM12878/GM12878_ep2vec_promoter.sent', 'r')
	promoter_test = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_promoter_test.sent', 'r')
	promoter_all = open('/home/ycm/data/GM12878/GM12878_ep2vec_promoter_all.sent', 'w')
	for line in promoter_train:
		promoter_all.write(line)
		promoter_all.write('\n')
	for line in promoter_test:
		promoter_all.write(line)
		promoter_all.write('\n')
	promoter_all.close()



#generateTraining: extract the training set from pairs.csv and output the training pair with sentence
def generateTraining():
	global enhancers_num,promoters_num,positive_num,negative_num
	fin1 = open('enhancers.bed','r')
	fin2 = open('promoters.bed','r')
	enhancers = []
	promoters = []
	for line in fin1:
		data = line.strip().split()
		enhancers.append(data[3])
		enhancers_num = enhancers_num + 1
	for line in fin2:
		data = line.strip().split()
		promoters.append(data[3])
		promoters_num = promoters_num + 1
	fin3 = open(cl+'train.csv','r')
	fout = open('training.txt','w')
	for line in fin3:
		if line[0] == 'b':
			continue
		else:
			data = line.strip().split(',')
			enhancer_index = enhancers.index(data[5])
			promoter_index = promoters.index(data[10])
			fout.write(str(enhancer_index)+'\t'+str(promoter_index)+'\t'+data[7]+'\n')
			if data[7] == '1':
				positive_num = positive_num + 1
			else:
				negative_num = negative_num + 1

# convert the sentence to doc2vec's tagged sentence

def doc2vec(name,k,vlen):
	filename  ='/home/ycm/data/GM12878/GM12878_ep2vec_promoter_all.sent'
	indexname = name.upper()
	sources = {filename:indexname}
	sentences = TaggedLineSentence(sources)
	model_1 = None
	model_1 = Doc2Vec(min_count=1, window=10, size=vlen, sample=1e-4, negative=5, workers=8)
	model_1.build_vocab(sentences.to_array())
	print(len(sentences.sentences_perm()))
	model_1.train(sentences.sentences_perm(), total_examples=len(sentences.sentences_perm()), epochs=10)
	model_1.save('/home/ycm/data/GM12878/GM12878_ep2vec_promoter'+'.d2v')

class TaggedLineSentence(object):
	def __init__(self, sources):
		self.sources = sources
		flipped={}
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

	def to_array(self):
		self.sentences = []
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences

	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

def train(k,vlen):


	global enhancers_num,promoters_num,positive_num,negative_num
	enhancer_test_index=76020
	promoter_test_index = 76020
	enhancer_model = Doc2Vec.load('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.d2v')
	promoter_model = Doc2Vec.load('/home/ycm/data/GM12878/GM12878_ep2vec_promoter.d2v')

	promoter_train = open('/home/ycm/data/GM12878/GM12878_ep2vec_promoter.sent')
	enhancer_train = open('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.sent')

	promoter_list_train = []
	enhancer_list_train = []
	enhancer_train_index=0
	promoter_train_index=0

	for enhancer in enhancer_train:

		enhancer = 'ENHANCERS_' + str(enhancer_train_index)
		enhancer_vec = enhancer_model.docvecs[enhancer]
		enhancer_list_train.append(enhancer_vec)

		enhancer_train_index =enhancer_train_index + 1

	for promoter in promoter_train:
		promoter = 'PROMOTERS_' + str(promoter_train_index)
		promoter_vec = promoter_model.docvecs[promoter]
		promoter_list_train.append(promoter_vec)

		promoter_train_index = promoter_train_index + 1

	promoter_train_np = numpy.array(promoter_list_train)
	enhancer_train_np = numpy.array(enhancer_list_train)

	print(promoter_train_np.shape)
	print(enhancer_train_np.shape)

	train_np = numpy.concatenate([enhancer_train_np, promoter_train_np],axis=1)

	print(train_np.shape)

	train_label = open('/home/ycm/data/GM12878/train/GM12878labels.txt')
	train_label_list = []
	for label in train_label:
		train_label_list.append(label)
	train_label_np = numpy.array(train_label_list).reshape((-1, 1))

	print(train_label_np)


	promoter_test = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_promoter_test.sent', 'r')
	enhancer_test = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_enhancer_test.sent', 'r')

	enhancer_list_test = []
	promoter_list_test = []
	for enhancer in enhancer_test:

		enhancer = 'ENHANCERS_' + str(enhancer_test_index)
		enhancer_vec = enhancer_model.docvecs[enhancer]
		enhancer_list_test.append(enhancer_vec)

		enhancer_test_index = enhancer_test_index + 1



	for promoter in promoter_test:

		promoter = 'PROMOTERS_' + str(promoter_test_index)
		promoter_vec = promoter_model.docvecs[promoter]
		promoter_list_test.append(promoter_vec)
		promoter_test_index = promoter_test_index + 1

	promoter_test_np = numpy.array(promoter_list_test)
	enhancer_test_np = numpy.array(enhancer_list_test)

	print(promoter_test_np.shape)
	print(enhancer_test_np.shape)

	test_np = numpy.concatenate([enhancer_test_np, promoter_test_np],axis=1)


	print(test_np.shape)

	test_label = open('/home/ycm/data/GM12878/test/GM12878labels.txt')
	test_label_list = []
	for label in test_label:
		test_label_list.append(label)
	test_label_np = numpy.array(test_label_list).reshape((-1, 1))

	print(test_label_np.shape)



	estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)

	estimator.fit(train_np, train_label_np)
	y_pre = estimator.predict(test_np)
	auc = metrics.roc_auc_score(test_label_np, y_pre)
	aupr = metrics.average_precision_score(test_label_np, y_pre)
	print(auc)
	print(aupr)

'''
	for line in fin:
		data = line.strip().split()
		prefix_enhancer = 'ENHANCERS_' + data[0]
		prefix_promoter = 'PROMOTERS_' + data[1]
		enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
		promoter_vec = promoter_model.docvecs[prefix_promoter]
		enhancer_vec = enhancer_vec.reshape((1,vlen))
		promoter_vec = promoter_vec.reshape((1,vlen))
		arrays[i] = numpy.column_stack((enhancer_vec,promoter_vec))
		labels[i] = int(data[2])
		i = i + 1
	cv = StratifiedKFold(y = labels, n_folds = 10, shuffle = True, random_state = 0)
	scores = cross_val_score(estimator, arrays, labels, scoring = 'f1', cv = cv, n_jobs = -1)
	print('f1:')
	print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	scores = cross_val_score(estimator, arrays, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
	print('auc:')
	print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	scores = cross_val_score(estimator, arrays, labels, scoring = 'average_precision', cv = cv, n_jobs = -1)
	print('auc:')
	print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

'''

enhancer_filename = '/home/ycm/data/GM12878/train/GM12878_enhancers.fasta'
promoter_filename = '/home/ycm/data/GM12878/train/GM12878_promoters.fasta'

#bed2sent(promoter_filename,6,1)
#bed2sent(enhancer_filename,6,1)
#print 'pre process done!'
#generateTraining()
#print 'generate training set done!'
#doc2vec("enhancers",6,100)
#doc2vec("promoters",6,100)
#print 'doc2vec done!'
train(6,100)
#genarare_file()
'''fin = open('/home/ycm/data/GM12878/test/GM12878_enhancers.fasta', 'r')
fout = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_enhancer_test.sent', 'w')
for line in fin:
	if line[0] == '>':
		continue
	else:
		line = line.strip().lower()
		length = len(line)
		i = 0
		while i <= length - 6:
			fout.write(line[i:i + 6] + ' ')
			i = i + 1
		fout.write('\n')
fin = open('/home/ycm/data/GM12878/test/GM12878_promoters.fasta', 'r')
fout = open('/home/ycm/data/GM12878/test/GM12878_ep2vec_promoter_test.sent', 'w')
for line in fin:
	if line[0] == '>':
		continue
	else:
		line = line.strip().lower()
		length = len(line)
		i = 0
		while i <= length - 6:
			fout.write(line[i:i + 6] + ' ')
			i = i + 1
		fout.write('\n')
'''
'''
enhancer_model = Doc2Vec.load('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.d2v')
#promoter_model = Doc2Vec.load('/home/ycm/data/GM12878/GM12878_ep2vec_promoter.d2v')

name = "enhancers"
filename  ='/home/ycm/data/GM12878/test/GM12878_ep2vec_enhancer_test.sent'
indexname = name.upper()
sources = {filename:indexname}
sentences = TaggedLineSentence(sources)
sentences.to_array()
enhancer_model.train(sentences.sentences_perm(), total_examples=len(sentences.sentences_perm()), epochs=10)
enhancer_model.save('/home/ycm/data/GM12878/GM12878_ep2vec_enhancer.d2v')
'''