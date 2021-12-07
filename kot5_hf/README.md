# KoT5(Pytorch)


##Setup

```
conda install cudnn==8.2.1
pip install -r requirements.txt
```


##Usage


###Tokenization

```
import torch
from transformer.models.tokenization_t5 import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(vocab_file)
df_train = pd.read_csv(dataset_list[0], sep='\t', error_bad_lines=False)
source_text = df_train['content'].values.tolist()
target_text = df_train['generate'].values.tolist()
tokenized_source_text = tokenizer(source_text, truncation=True, padding='max_length', max_length=512)
tokenized_target_text = tokenizer(target_text, truncation=True, padding='max_length', max_length=512)
```

###Model

```
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
model = T5ForConditionalGeneration.from_pretrained(model_file, local_files_only=True)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=0.001,
    evaluation_strategy='steps', # Run evaluation every eval_steps
    save_steps=1000, # How often to save a checkpoint
    remove_unused_columns=True, # Removes useless columns from the dataset
    run_name='run_name', # Wandb run name
    logging_steps=1000, # How often to log loss to wandb
    eval_steps=1, # How often to run evaluation on the val_set
    metric_for_best_model='rouge1_fmeasure', # Use loss to evaluate best model.
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset=training_set,
    eval_dataset=validation_set
)
```

###Train

```
if do_train==True :
    trainer.train()
    trainer.save_model(output_dir + '/model')
```

###Evaluate

```
if do_eval==True :
    metrics=trainer.evaluate()
    print(metrics)
```

###Predict

```
if do_predict==True :
    model_dir = output_dir + '/model'
    output_dir = output_dir + '/model/eval'

    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    import torch

    print("input:")
    input_text = input()

    with torch.no_grad():
        tokenized_text = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')

        source_ids = tokenized_text['input_ids']
        source_mask = tokenized_text['attention_mask']

        generated_ids = model.generate(
            input_ids = source_ids,
            attention_mask = source_mask,
            max_length=512,
            num_beams=5,
            repetition_penalty=1,
            length_penalty=1,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("\noutput:\n" + pred)
```