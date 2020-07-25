from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import torch
model = torch.load("1")  



def test_sententence(tex,max_words,max_len):
	token = Tokenizer(num_words=max_words)
	print(" FINDING ACCURACY FOR THE MODEL WHICH HAS EPOCHS: 1")
	model = torch.load("1")  
	seq_tex = token.texts_to_sequences(tex)
	matrix_tex = sequence.pad_sequences(seq_tex,maxlen=max_len)
	y_pre=model.predict(matrix_tex)
	if y_pre>=0.5:
	  y_pre=1
	else:
	  y_pre=0
	return y_pre




