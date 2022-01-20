# How to train a new language model from scratch using Transformers and Tokenizers
# Copyright 2020, Denis Rothman. Denis Rothman adapted a Hugging Face reference notebook to pretrain a transformer model.
# The next steps would be work on the building a larger dataset and testing several transformer models.
#
# The Transformer model of this Notebook is a Transformer model named ***KantaiBERT***. ***KantaiBERT*** is trained
# as a RoBERTa Transformer with DistilBERT architecture. The dataset was compiled with three books by Immanuel Kant
# downloaded from the [Gutenberg Project](https://www.gutenberg.org/).
#
# <img src="https://eco-ai-horizons.com/data/Kant.jpg" style="margin: auto; display: block; width: 260px;">
#
# ![](https://commons.wikimedia.org/wiki/Kant_gemaelde_1.jpg)
#
# ***KantaiBERT*** was pretrained with a small model of 84 million parameters using the same number of layers and heads
# as DistilBert, i.e., 6 layers, 768 hidden size,and 12 attention heads. ***KantaiBERT*** is then fine-tuned for a
# downstream masked Language Modeling task.
#
# ### The Hugging Face original Reference and notes:
# Notebook edition (link to original of the reference blogpost [link](https://huggingface.co/blog/how-to-train)).


# #@title Step 2:Installing Hugging Face Transformers
# # We won't need TensorFlow here
# get_ipython().system('pip uninstall -y tensorflow')
# # Install `transformers` from master
# get_ipython().system('pip install git+https://github.com/huggingface/transformers')
# get_ipython().system("pip list | grep -E 'transformers|tokenizers'")
# # transformers version at notebook update --- 2.9.1
# # tokenizers version at notebook update --- 0.7.0

# @title Step 3: Training a Tokenizer
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("**/*.txt")]
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(
    files=paths,
    vocab_size=52_000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)


# @title Step 4: Saving the files to disk
import os

token_dir = "/content/KantaiBERT"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
tokenizer.save_model("KantaiBERT")


# @title Step 5 Loading the Trained Tokenizer Files
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "./KantaiBERT/vocab.json",
    "./KantaiBERT/merges.txt",
)
print(tokenizer.encode("The Critique of Pure Reason.").tokens)

tokenizer.encode("The Critique of Pure Reason.")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# @title Step 6: Checking Resource Constraints: GPU and NVIDIA
# get_ipython().system('nvidia-smi')

# @title Checking that PyTorch Sees CUDAnot
import torch

torch.cuda.is_available()

# @title Step 7: Defining the configuration of the Model
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


print(config)


# @title Step 8: Re-creating the Tokenizer in Transformers
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("./KantaiBERT", max_length=512)


# @title Step 9: Initializing a Model From Scratch
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model)


print(model.num_parameters())


# @title Exploring the Parameters
LP = list(model.parameters())
lp = len(LP)
print(lp)
for p in range(0, lp):
    print(LP[p])


# @title Counting the parameters
np = 0
for p in range(0, lp):  # number of tensors
    PL2 = True
    try:
        L2 = len(LP[p][0])  # check if 2D
    except:
        L2 = 1  # not 2D but 1D
        PL2 = False
    L1 = len(LP[p])
    L3 = L1 * L2
    np += L3  # number of parameters per tensor
    if PL2 == True:
        print(p, L1, L2, L3)  # displaying the sizes of the parameters
    if PL2 == False:
        print(p, L1, L3)  # displaying the sizes of the parameters

print(np)  # total number of parameters


# @title Step 10: Building the Dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kant.txt",
    block_size=128,
)


# @title Step 11: Defining a Data Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# @title Step 12: Initializing the Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


# @title Step 13: Pre-training the Model
trainer.train()


# @title Step 14: Saving the Final Model(+tokenizer + config) to disk
trainer.save_model("./KantaiBERT")


# @title Step 15: Language Modeling with the FillMaskPipeline
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="./KantaiBERT", tokenizer="./KantaiBERT")


fill_mask("Human thinking involves<mask>.")
