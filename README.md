# -DLNLP_assignment_24-

The imports required to run the code include evaluate, numpy, datasets, transformers and matplotlib.

In main.py the required imports are already in the code, and is shown again here:
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

Line 226 and 227 in main.py contains evaluations to the test dataset, where line 226 is the finetuned model and 227 the pretrained model. By default, the finetuned evaluation will run the results are shown in terminal.

Lines 259 to 266 of main.py contains baseline comparisons using ner pipeline and can be commented out to see the results of baseline pipeline methods.  The 'test_text_1' in line 261 can be changed to 'test_text_0' to test the other text.
Alternatively, you can add text variables and feed in line 261 to check baseline performances.


Lines 269 and 270 are tests using token classification pipeline which can be uncommented to see the result (recommend only uncommenting 270 as the output is shorter).
Again, the variable can be changed to test_text_1 if desired
