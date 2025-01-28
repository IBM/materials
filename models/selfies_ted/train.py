from transformers import BartConfig
from transformers import BartForConditionalGeneration
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import numpy as np
import torch, argparse
import evaluate
import feather
from datasets import Dataset, load_dataset

# Train SELFIES-TED(small)

def train(training_dataset_path: str, base_model_path: str, epochs: int):

    if training_dataset_path==None:
        training_dataset_path = "./H22M400.ftr"

    print("Training dataset: ", training_dataset_path)
    if training_dataset_path.endswith(".ftr"):
        dataset = Dataset.from_pandas(feather.read_dataframe(training_dataset_path))
    else:
        dataset = load_dataset("csv", data_files=training_dataset_path)["train"]

    print("Number of samples : ", len(dataset))

    if base_model_path==None:
        print("Training from scratch")
        base_model_path = "ibm/materials.selfies-ted2m"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        token2idx = tokenizer.get_vocab()
        config = BartConfig()
        config.vocab_size = len(token2idx)
        config.pad_token_id = token2idx["[PAD]"]
        config.eos_token_id = token2idx["[SEP]"]
        config.bos_token_id = token2idx["[CLS]"]

        ### config changes (2.2M param)

        config.encoder_ffn_dim = 256
        config.decoder_ffn_dim = 256

        config.encoder_attention_heads = 4
        config.decoder_attention_heads = 4

        config.encoder_layers = 2
        config.decoder_layers = 2

        config.num_hidden_layers = 6
        config.max_position_embeddings = 128
        config.dmodel = 256
        config.d_model = 256

        model = BartForConditionalGeneration(config)
    else:
        print("Loading base model ", base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = BartForConditionalGeneration.from_pretrained(base_model_path)    
    

    def preprocess_function(examples):
        return tokenizer([x for x in examples["SELFIES"]], return_tensors='pt', max_length=128, truncation=True,
                         padding='max_length')

    tokenized_inputs = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Total parameters : ", sum(p.numel() for p in model.parameters()))

    def make_labels(examples):
        del examples["token_type_ids"]
        examples["labels"] = examples["input_ids"].copy()


        rand = torch.rand(torch.Tensor(examples["input_ids"]).shape)
        mask = (rand < 0.15) * (torch.Tensor(examples["input_ids"]) != tokenizer.pad_token_id) * (
                    torch.Tensor(examples["input_ids"]) != tokenizer.cls_token_id) * (
                       torch.Tensor(examples["input_ids"]) != tokenizer.sep_token_id)

        masked_input_ids = torch.where(mask, tokenizer.mask_token_id, torch.Tensor(examples["input_ids"]))

        examples["input_ids"] = masked_input_ids.tolist()

        return examples

    lm_dataset = tokenized_inputs.map(make_labels, batched=True, num_proc=16)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

        labels = [labels[row][indices[row]] for row in range(len(labels))]
        labels = [item for sublist in labels for item in sublist]

        predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
        predictions = [item for sublist in predictions for item in sublist]

        results = metric.compute(predictions=predictions, references=labels)
        results["eval_accuracy"] = results["accuracy"]
        print(results)
        results.pop("accuracy")

        return results

    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="no",
        num_train_epochs=epochs,
        per_device_train_batch_size=256,
        logging_strategy="epoch",
        save_strategy="epoch",
        do_eval=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Training Completed! ")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SELFIES-TED training')
    parser.add_argument('-f', '--file', required=False, type=str)
    parser.add_argument('-b', '--base-model', required=False, type=str)
    parser.add_argument('-e', '--epochs', required=False, type=int, default=1)
    args = parser.parse_args()
    print(args)
    train(args.file, args.base_model, args.epochs)
    print("Completed!")