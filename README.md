# -DLNLP_assignment_24-

The imports required to run the code include evaluate, numpy, datasets, transformers and matplotlib.

main.py contains the code that requires the imports, and is shown here:
"""
import evaluate                                 #import required modules
import numpy as np                          
from datasets import load_dataset           #for dataset loading
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, pipeline
import matplotlib.pyplot as plt
"""

By default, main.py downloads the conll2003 dataset once and can be accessed in cache after intial loading.


In line 43 of main.py, the showhistogram function can be uncommented to show the histogram of the dataset row lengths. Just close the histogram after viewing it, then the rest of the code will continue running.


!CAUTION
In line 217 of main.py uncommenting this WILL TRAIN A NEW MODEL, so do not train a new model unless required as it will overide the trained model in 'ner-finetuned-model'
!CAUTION


In line 218 of main.py, this stores the directory of the finetuned model. To run the code properly, the directory needs to be changed to the directory where you downloaded the model:
"Your directories\-DLNLP_assignment_24-\ner-finetuned-model\checkpoint-5268" 
An example is shown here:   finetuned = r"C:\Users\User\Documents\Github\-DLNLP_assignment_24-\ner-finetuned-model\checkpoint-5268" 


Line 226 and 227 in main.py contains evaluations to the test dataset, where line 226 is the finetuned model and 227 the pretrained model. By default, the finetuned evaluation will run the results are shown in terminal.
Line 227 can be uncommented to see the evaluation results the default model. When both lines are uncommented, the evaluation will run twice and the results for both models will display afterwards.

Lines 258 to 265 of main.py contains baseline comparison using ner pipeline and can be commented out to see the results on test text (this is not the test dataset but just real life text).  The 'test_text_1' in line 261 can be changed to 'test_text_0' to test the other real life text.
Alternatively, you can add text variables and feed in line 261 to check baseline performances.


Line 268 is the baseline pipeline result of token classificaiton on pre-trained model, variable can be change to test_text_1 if desired.
Line 269 is the finetuned pipeline result token classification pipeline, variable can be changed to test_text_1 if desired. 
Again, new text variables can be fed in as well.


The overall expected output when the code is running properly without changing anything but the location of the model:

Data structure: DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3453
    })
})

List of all tags: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
Corrsponding tag integer to tag string: {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

Sample sentence: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
NER class in integer: [3, 0, 7, 0, 0, 0, 7, 0, 0]
NER class in string: ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']


Tokenized output: ['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
Consecutive word ids can be seen:  [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]

Hence needed alignment.

tag_int after alignment and padding: [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]


100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 432/432 [00:08<00:00, 52.56it/s] 

Test Results: {'eval_loss': 0.18351727724075317, 'eval_precision': 0.8894048847609219, 'eval_recall': 0.9155453257790368, 'eval_f1': 0.9022858139940675, 'eval_accuracy': 0.9727549203447787, 'eval_runtime': 8.3509, 'eval_samples_per_second': 413.49, 'eval_steps_per_second': 51.731}

