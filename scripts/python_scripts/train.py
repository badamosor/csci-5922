"""Training
==========

reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from torch import optim
import torch.nn as nn
from helper import *
from models import *

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attn, max_length=50):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)


    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        if attn:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, input_lang, output_lang, train_pairs, n_iters, attn, char_based, print_every=1000, plot_every=100, learning_rate=0.01, save_every_iters= 10):
    start = time.time()
    averageLoss = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
        
    for i in range(n_iters):
      print(i)
      numExample = 0
      for pairs in train_pairs:
          training_pair = tensorsFromPair(pairs, input_lang, output_lang, char_based)
          input_tensor = training_pair[0]
          target_tensor = training_pair[1]

          loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion, attn)
          print_loss_total += loss
          plot_loss_total += loss

          numExample += 1

          if (numExample % print_every == 0):
              print_loss_avg = print_loss_total / print_every
              print_loss_total = 0
              print('%s %d %.4f %d' % (timeSince(start, numExample), numExample, print_loss_avg, i))

      #accuracy = getAccuracy(encoder1, decoder)
      #accuracies.append(accuracy)
      averageLoss.append(print_loss_total / print_every)
      print(('%.4f,%.4f') % ((print_loss_total / print_every),loss))
          #print(accuracy)
