"""Evaluation
==========

Evaluation is mostly the same as training, but there are no targets so
we simply feed the decoder's predictions back to itself for each step.
Every time it predicts a word we add it to the output string, and if it
predicts the EOS token we stop there. We also store the decoder's
attention outputs for display later.

reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from models import *
from tokenization import megasplit


def evaluate(encoder, decoder, input_lang, output_lang, sentence, attn, char_based, max_length=50):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, char_based)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):

            if (attn):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)            
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
         
        if (attn):
            return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def evaluateDiacriticGenerator(encoder, decoder, input_lang, output_lang, test_pairs, attn, char_based):
    correct_sentence_pair = 0
    incorrect_sentence_pair = 0

    correct_word_pair = 0
    incorrect_word_pair = 0

    correct_diac = 0
    incorrect_diac = 0

    regex = "[\u0618-\u061A|\u064B-\u0653]"

    count = 0
    sentenceInput = []
    sentenceTarget = []
    sentenceOutput = []

    
    for pair in test_pairs:
        if char_based:
            if (attn):
                output_chars, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0], attn, char_based)
            else:
                output_chars = evaluate(encoder, decoder, input_lang, output_lang, pair[0], attn, char_based)
            output_word = ''.join(output_chars)

            if (count < 4):
                sentenceInput.append(pair[0])
                sentenceTarget.append(pair[1])
                sentenceOutput.append(output_word)
                count += 1
            else:
                #Accuracy on sentence basis

                if (sentenceTarget == sentenceOutput):
                    correct_sentence_pair += 1
                    print('correct: ',sentenceTarget, sentenceOutput)

                else:
                    incorrect_sentence_pair += 1
                    print('incorrect: ',sentenceTarget, sentenceOutput)

                sentenceInput = []
                sentenceTarget = []
                sentenceOutput = []

                sentenceInput.append(pair[0])
                sentenceTarget.append(pair[1])
                sentenceOutput.append(output_word)
                count = 1

            #Accuracy on word basis
            if (pair[1] == output_word):
                if (re.search(regex, pair[1])):
                    correct_diac += 1
                    correct_word_pair += 1
            else:
                if (re.search(regex, pair[1])):
                    incorrect_diac += 1
                    incorrect_word_pair += 1


        else:
            if (attn):
                output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0], attn, char_based)
            else:
                output_words = evaluate(encoder, decoder, input_lang, output_lang, pair[0], attn, char_based)
            output_sentence = ' '.join(output_words)
            
            #Accuracy on sentence basis
            if (pair[1] == output_sentence):
                correct_sentence_pair += 1
            else:
                incorrect_sentence_pair += 1

            #Accuracy on word basis
            tokenized_target = megasplit(' ', pair[1])

            for target, output in zip(tokenized_target, output_words):
                if (target == output):
                    correct_word_pair += 1
                else:
                    incorrect_word_pair += 1
    
    print("accuracy of words:",correct_word_pair/(correct_word_pair+incorrect_word_pair),correct_word_pair,incorrect_word_pair)
    print("Number of all words: ",correct_word_pair+correct_word_pair)

    print("accuracy of sentence:",correct_sentence_pair/(correct_sentence_pair+incorrect_sentence_pair),correct_sentence_pair,incorrect_sentence_pair)
    print("Number of all sentences: ", correct_sentence_pair+incorrect_sentence_pair)
