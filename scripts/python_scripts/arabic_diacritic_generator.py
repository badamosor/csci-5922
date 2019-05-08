from prepareData import prepareData
from models import *
from train import *
from test import *

#Set parameters

char_based = True
attn = False
n_iter = 60
hidden_size = 256

#Prepare data
if (char_based):
    input_lang, output_lang, train_pairs, test_pairs = prepareData('NoDiac', 'Diac', '../../dataset_for_train_test/char_basedTrain.data', '../../dataset_for_train_test/char_basedTest.data', char_based)

else:
    input_lang, output_lang, train_pairs, test_pairs = prepareData('NoDiac', 'Diac', '../../dataset_for_train_test/trainig_data_3000.csv', '../../dataset_for_train_test/test_data_3000_425.csv', char_based)

#Encoder
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

#Decoder
if (attn):
    decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
else:
    decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

#Train
trainIters(encoder1, decoder1, input_lang, output_lang, train_pairs, n_iter, attn, char_based, print_every=1000)

#Test
evaluateDiacriticGenerator(encoder1, decoder1, input_lang, output_lang, test_pairs, attn, char_based)
