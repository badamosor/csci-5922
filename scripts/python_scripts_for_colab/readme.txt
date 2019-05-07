arabic_seq2seq_attn_char_based.py: Character-based seq2seq file
arabic_seq2seq_trans.py: Word-based seq2seq file

In order to run these file

1. Open google colab
2. Upload datasets from dataset_for_train_test
3. run following in the colab cell and choose one of above files

from google.colab import files
src = list(files.upload().values())[0]
open('diacritic_generator.py','wb').write(src)
import diacritic_generator
