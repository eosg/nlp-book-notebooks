from json.tool import main
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
)
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import evaluate

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def data_preprocess():
    print("begin data preprocess")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader


def train_model(train_dataloader, eval_dataloader):
    print("begin train model")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(f"{num_training_steps=}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # device
    device(type="cuda")
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    return model


def eval(model):
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(f"{metric.compute()=}")


if __name__ == "__main__":
    train_dataloader, eval_dataloader = data_preprocess()
    model = train_model(train_dataloader, eval_dataloader)
    eval(model)
