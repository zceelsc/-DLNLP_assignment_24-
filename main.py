import evaluate                                 #import required modules
import numpy as np                          
from datasets import load_dataset           #for dataset loading
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, pipeline
import matplotlib.pyplot as plt


#loading data
data = load_dataset("conll2003")  #downloads and loads the dataset which contains train, validation and test sets
print('Data structure:', data)    #show training, validation and test set as well as keys
print()

# Access train, validation, and test sets
train_set = data['train']
val_set = data['validation']
test_set = data['test']

# Function to calculate length of each row
def get_row_lengths(dataset):
    row_lengths = [len(row) for row in dataset['tokens']]
    return row_lengths

def showhistogram():
  # Calculate row lengths for train, val, and test sets
  train_row_lengths = get_row_lengths(train_set)
  val_row_lengths = get_row_lengths(val_set)
  test_row_lengths = get_row_lengths(test_set)

  # Plot histograms
  plt.figure(figsize=(10, 6))

  plt.hist(train_row_lengths, bins=30, alpha=0.5, edgecolor='black', label='Train Set')
  plt.hist(val_row_lengths, bins=30, alpha=0.5, edgecolor='black', label='Validation Set')
  plt.hist(test_row_lengths, bins=30, alpha=0.5, edgecolor='black', label='Test Set')
  plt.xlabel('Length of Rows')
  plt.ylabel('Frequency')
  plt.title('Histogram of Row Lengths in Conll2003 Dataset')
  plt.legend()

  plt.show()

#showhistogram()   # shows lengths of tokens per row

all_tag_int = data['train'].features['ner_tags'].feature    #get ner_tags (integers)

def link_int2str(batch):   #function for attaching names from indexes e.g. 3 -> B-ORG
  ner_tag_str = {'ner_tag_str': [all_tag_int.int2str(idx) for idx in batch['ner_tags']]}    #get ner_tags, converts into tag name using int2str
  return ner_tag_str

data = data.map(link_int2str)        #maps the data with the created function




ner_feature = data['train'].features['ner_tags']   #get features of NER data
tag_str_spectrum = ner_feature.feature.names       #get the tag spectrum in string
print('List of all tags:',tag_str_spectrum)

id2label = {tag_int:tag_str for tag_int, tag_str in enumerate(tag_str_spectrum)}  #linking tag_int to tag_str  e.g.  0: '0', 1: 'B-PER' Created for model training
label2id = {tag_str:tag_int for tag_int, tag_str in enumerate(tag_str_spectrum)}  #str2int                                              Created for model training
print('Corrsponding tag integer to tag string:', id2label)


inputs = data['train'][0]['tokens']                             #extract for example
print()
print('Sample sentence:', inputs)

tag_int_row = data['train'][0]['ner_tags']                      #get example tag_int  e.g. '3,0,7,0,0'
print('NER class in integer:',  tag_int_row)                    #show the example

tag_str_row = data['train'][0]['ner_tag_str']                   #get tag in string form
print('NER class in string:', tag_str_row)                      #show the strings

print()




model_checkpoint = "bert-base-cased"                            #loads model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)     #load tokenizer

inputs = tokenizer(inputs, is_split_into_words=True)            #tokenize the example
print('Tokenized output:', inputs.tokens())                     #shows the misalignment problem

word_ids = inputs.word_ids()                                    #get example word_ids
print('Consecutive word ids can be seen: ',word_ids)                                    #show the word_ids duplication
print()
print('Hence needed alignment.')


def fix_misalignment(tag_int_row, word_ids):   #function for aligning tags and tokens
  prev_id = None               #temporary variable
  aligned_ints = []            #create list for storing aligned tag ints
  padding_value = -100
  #loop for each word in word_ids
  for current_id in word_ids:   

    #non-duplicate cases
    if current_id != prev_id:       #if (current word_id) != (previous word_id)

      prev_id = current_id           #update current_id into prev_id

      #Since prev_id updates per iteration, current_id will have instances = None, 
      #hence, set tag_int to -100 if current_id=None, while others set according to tag_int_row.
      if current_id == None:
        tag_int = padding_value
      else:
        tag_int = tag_int_row[current_id]

      aligned_ints.append(tag_int)     #add correct tag_int into aligned_ints

    elif current_id == None:       #when word id is None e.g. BOS EOS, set to padding
      tag_int = padding_value            # use padding value
      aligned_ints.append(tag_int)   #append to list

    #duplicate cases (e.g. 7,7)
    else:   #case when current_id = prev_id (same word_id) (one word split into two tokens)
      tag_int = tag_int_row[current_id]
      
      #Since we dont want the same word to have two beginning tokens e.g. (B-xxx, B-xxx), 
      #and since B-xxx in feature names are always odd number, hence turn the latter
      # to I-xxx by adding 1 to the tag_int, resulting in (B-xxx, I-xxx)
      if tag_int % 2 != 0:           
        tag_int += 1

      aligned_ints.append(tag_int)    #append to list

  return aligned_ints

print()
print('tag_int after alignment and padding:', fix_misalignment(tag_int_row, word_ids))  #apply function and show result for sentence
print()

def tokenize_and_align(datas):       #function to tokenize and align labels
  
  aligned_ints_complete = []                            #set empty list for aligned tag_ints to store

  tokenized_inputs = tokenizer(datas['tokens'], truncation=True, is_split_into_words=True)  #tokenize the data

  complete_tag_ints = datas['ner_tags']                 #get all tag_ints

  for i, tag_int_row in enumerate(complete_tag_ints):   #loop for tokenized inputs and all tag ints

    word_ids = tokenized_inputs.word_ids(i)             #get word_ids of tokenized inputs

    aligned_ints_complete.append(fix_misalignment(tag_int_row, word_ids))  #append aligned_int list

  tokenized_inputs['labels'] = aligned_ints_complete   #stored in label key

  return tokenized_inputs
#map using function and clear unused columns for processing speed
print('Mapping the data:')
tokenized_datasets = data.map(tokenize_and_align, batched=True, remove_columns=data['train'].column_names)
print()


#use datacollator for auto padding purposes
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


metric = evaluate.load('seqeval')      #load metric for calculations

#function for computing trainning and eval metrics
def compute_metrics(preds_and_true_tag_ints):                #inputs are model predictions and all tag_int
  logits, all_true_tag_int = preds_and_true_tag_ints         #unpack into logits and true tag_ints
  predictions = np.argmax(logits, axis=-1)                   #return index of predicted class

  true_tag_str = [[tag_str_spectrum[tag_int] for tag_int in tag_int_row if tag_int!=-100] for tag_int_row in all_true_tag_int]   #gets true tag_str for all rows, converted from all tag_ints, as long as ints are not -100
  predicted_tag_str = [[tag_str_spectrum[pred] for pred,tag_int in zip(prediction, tag_int_row) if tag_int!=-100] for prediction, tag_int_row in zip(predictions, all_true_tag_int)]  #get predicted tag_str, ignoring -100

  result_metrics = metric.compute(predictions=predicted_tag_str, references=true_tag_str)         #compute the metrics

  return {"precision": result_metrics['overall_precision'],"recall": result_metrics['overall_recall'],                              #retrun the metrics
          "f1": result_metrics['overall_f1'],"accuracy": result_metrics['overall_accuracy']}





model_for_train = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)  #model used to train

#training arguments
args = TrainingArguments("ner-finetuned-model",   #name for storing
                        evaluation_strategy = "epoch", #get eval on epoch
                        save_strategy="epoch",         #Save on every epoch
                        learning_rate = 2e-5,          #learning rate 
                        num_train_epochs=3,            #no. of epoch
                        weight_decay=0.01)             #adamW weight decay

def model_training(model_for_train):
    #initialise trainer
    trainer = Trainer(model=model_for_train,
                    args=args,                            #get args from above
                    train_dataset = tokenized_datasets['train'],  #load the trainning data
                    eval_dataset = tokenized_datasets['validation'],  #load the validation data
                    data_collator=data_collator,              #use datacollator
                    compute_metrics=compute_metrics,          #use metrics function
                    tokenizer=tokenizer)                      #use autotokenizer

    trainer.train()                 #activate training

def evaluation(applied_model):
    trainer = Trainer(model=applied_model,                 #apply the model for evaluation
                    data_collator=data_collator,              #use datacollator
                    compute_metrics=compute_metrics          #use metrics function
                    )    

    test_dataset=tokenized_datasets['test']             #use the test dataset
    test_results = trainer.evaluate(eval_dataset=test_dataset)      #evaluate
    print()
    print('Test Results:', test_results)            #print results
    print()



#model_training(model_for_train)               #!!uncommenting this will train a new model
finetuned = r"C:\Users\User\DLNLP\ner-finetuned-model\checkpoint-5268"            #tuned model's file path



finetuned_eval = AutoModelForTokenClassification.from_pretrained(finetuned)                #finetuned model for evaluation
default_eval= AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)    #not tuned model (default)


evaluation(finetuned_eval)            #eval on tuned model
#evaluation(default_eval)             #eval on untuned model


#additional test for any text
token_classifier_default = pipeline("token-classification", model=default_eval, tokenizer=tokenizer, aggregation_strategy="simple")
token_classifier_tuned=pipeline("token-classification", model=finetuned_eval, tokenizer=tokenizer, aggregation_strategy="simple")




test_text_0="Look at the leap Haaland's got, it's unplayable, it's Unstoppable, look how high he gets, its Cristiano Ronaldo like"
test_text_1="""Reporter: we appreciate you coming to talk to us, so close after the final whistle, first of all, the end to that game, the Bukayo
Saka penalty shout, what's your initial thoughts? Trossard: yeah on the pitch, it looked like a penalty to me, I haven't seen the
replay, but uh yeah for me it looked like it was clear contact, so but I have to wait after to to see it again obviously.
Reporter: aside from that moment being behind at half time and be able being able to come back and get a draw out of this game,
how are you feeling as a group, are you pleased with that results? Trossard: I think not pleased but obviously when you're 2-1 down at
halftime, you take a point at the end. Obviously I think we started so well the the first half, the first 15 20
minutes were so good and then yeah we could have scored two or three, I think and then yeah when they score The
Equalizer, they yeah you can see how what what kind of quality they have to hurt us as well and yeah at the end of the day
it's a draw so we take it and then we have to finish it off when we go there. Reporter: when you came on, what was the message
from the manager to you, what did he ask of you? Trossard: yeah had to try to turn it around and uh luckily I help the team again
with the with a goal so I'm pleased with that and yeah now we have to work hard for the next game obviously. Reporter: you've
scored more goals as a substitute than any other player in the top five leagues in Europe, that's quite something you're
a Difference Maker. Trossard: yeah I always try to do yeah what whatever I can on the pitch to to help
the team if I start or if I come on as a sub uh and yeah that's that's a great stat so hopefully I can help a bit more
in the in the next weeks. Reporter: there was so many good battles out there on the pitch going into the second leg in Munich now
how much belief do you have that you can win this and get through to the next stage? Trossard: yeah still a a good belief I think
um we have such a great team and um we know we'll be hard there as well but uh if we play at at our best
level I think we can beat anyone and we have to show it there and um like I said it's it's a tough schedule coming up so
uh we need to recover now and uh then play the game in the weekend. Reporter: well play tonight. Trossard: thank you"""

#base line comparison 1
"""
ner_pipeline = pipeline("ner")
ner_results = ner_pipeline(test_text_1)

# Display NER results
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']}")
"""


#print(token_classifier_default(test_text_0))    #result on test text 0
#print(token_classifier_tuned(test_text_1))     #result on test text 1

