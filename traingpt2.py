from readdata import read_text_from_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset


def train_gpt2(file_name:str):
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    content = ""

    try:
        with open(file_name, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    
    tokenized_data = tokenizer(content, return_tensors="pt", truncation=True, padding=True)
    data = [{"input_ids":tokenized_data["input_ids"]}]
    dataset = Dataset.from_list(data)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir="./trainedmodel",
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model("./trainedmodelfinetuned")

    return tokenizer, model

if __name__ == "__main__":
    train_gpt2("policies.txt")

    
# def tokenize_function(examples, tokenizer, block_size=128):
#     """
#     Tokenizes a batch of texts with sliding window for causal LM.
#     """
#     # Flatten list of texts if needed
#     texts = examples["text"]
#     if not texts or all(not t.strip() for t in texts):
#         return {"input_ids": [], "labels": []}  # Empty batch, will be skipped
    
#     tokenized = tokenizer(
#         texts,
#         truncation=False,
#         padding=False,
#         return_overflowing_tokens=True,
#         max_length=block_size,
#         stride=(block_size * 2 // 3)  # ~66% overlap
#     )
    
#     # Handle case where encodings is None (empty inputs)
#     if tokenized.encodings is None:
#         return {"input_ids": [], "labels": []}
    
#     # Extract overflowing sequences
#     input_ids_list = [encoding.ids for encoding in tokenized.encodings]  # Use .ids directly
    
#     return {
#         "input_ids": input_ids_list,
#         "labels": input_ids_list  # Copy for causal LM
#     }

# def group_texts(examples, block_size=128):
#     if not examples["input_ids"]:
#         return examples  # Empty, pass through
    
#     concatenated_examples = {
#         "input_ids": sum(examples["input_ids"], []),
#         "labels": sum(examples["labels"], [])
#     }
    
#     total_length = len(concatenated_examples["input_ids"])
#     if total_length == 0:
#         return {"input_ids": [], "labels": []}
    
#     # Truncate to multiple of block_size
#     total_length = (total_length // block_size) * block_size
#     for key in concatenated_examples:
#         concatenated_examples[key] = concatenated_examples[key][:total_length]
    
#     # Re-chunk
#     result = {
#         key: [t[i:i + block_size] for i in range(0, len(t), block_size)]
#         for key, t in concatenated_examples.items()
#     }
    
#     return result

# def train_gpt2(policies_folder:str, output_dir: str = "./gpt2-finetuned-hr", block_size: int = 128):
#     documents = []

#     for filename in os.listdir(policies_folder):
#         file_path = os.path.join(policies_folder, filename)
#         if os.path.isfile(file_path):
#             try:
#                 text = read_text_from_file(file_path)
#                 documents.append(text)
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
    
#     if not documents:
#         raise ValueError("No valid documents found in the folder.")
    
#     training_text = "\n\n".join(documents)
    
#     dataset_dict = {"text": [training_text]} 
#     full_dataset = Dataset.from_dict(dataset_dict)
    
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#     tokenizer.pad_token = tokenizer.eos_token

#     tokenized_dataset = full_dataset.map(
#         lambda examples: tokenize_function(examples, tokenizer, block_size),
#         batched=True,
#         batch_size=1000,
#         remove_columns=full_dataset.column_names  # Remove text column for efficiency
#     )

#     tokenized_dataset = tokenized_dataset.filter(lambda ex: len(ex["input_ids"]) > 0)
    
#     if len(tokenized_dataset) == 0:
#         raise ValueError("No valid tokenized sequences after processing. Text too short?")
    
#     # Group for batching
#     grouped_dataset = tokenized_dataset.map(
#         lambda ex: group_texts(ex, block_size),
#         batched=True,
#         batch_size=1000,
#         remove_columns=tokenized_dataset.column_names
#     )
    
#     # Flatten the grouped lists back to individual examples
#     def flatten(example):
#         return {
#             "input_ids": [token for sublist in example["input_ids"] for token in sublist],
#             "labels": [token for sublist in example["labels"] for token in sublist]
#         }
    
#     grouped_dataset = grouped_dataset.map(flatten, batched=False)
#     grouped_dataset = grouped_dataset.filter(lambda ex: len(ex["input_ids"]) >= block_size)
    
#     # Split train/eval
#     train_test_split = grouped_dataset.train_test_split(test_size=0.1, seed=42)
#     train_dataset = train_test_split["train"]
#     eval_dataset = train_test_split["test"]
    
#     # Data collator
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#         pad_to_multiple_of=8
#     )
    
#     # Load model
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
    
#     # Training args
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         evaluation_strategy="epoch" if len(eval_dataset) > 0 else "no",
#         learning_rate=2e-5,
#         save_steps=1000,
#         save_total_limit=2,
#         logging_steps=100,
#         report_to="none",
#         fp16=True,  # If GPU available
#         prediction_loss_only=True
#     )
    
#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#     )
    
#     # Train and save
#     trainer.train()
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
    
#     print(f"Model fine-tuned and saved to {output_dir}. Dataset size: {len(train_dataset)} train examples.")
#     return tokenizer, model



