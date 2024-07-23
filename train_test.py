from unsloth import FastLanguageModel
import torch
import json

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import json
from datasets import Dataset


max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Preparation of the dataset for training
def convert_multiple_pairs_to_json(text_file, json_file, pair_delimiter="===", response_delimiter="---"):
    try:
        # Read the entire text file content
        with open(text_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split the content into multiple pairs based on the pair delimiter
        pairs = content.split(pair_delimiter)

        # Initialize a list to store all the input-response pairs
        json_content = []

        # Process each pair separately
        for pair in pairs:
            if response_delimiter in pair:
                # Split the pair into input and response
                input_text, response_text = pair.split(response_delimiter, 1)

                # Strip whitespace and add to the list
                json_content.append({
                    "input": input_text.strip(),
                    "output": response_text.strip()
                })

        # Write the list of dictionaries to a JSON file
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(json_content, file, ensure_ascii=False, indent=4)

        print(f"Successfully converted {text_file} to {json_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the input text file and output JSON file
text_file = 'Dataset.txt'  
json_file = 'input_response_pairs.json' 

# Convert the input-response pairs to JSON
convert_multiple_pairs_to_json(text_file, json_file)



# Load JSON dataset
def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_dataset(data):
    # Change 'input' and 'response' to match your data keys
    inputs = [item['input'] for item in data]
    responses = [item['output'] for item in data]
    return Dataset.from_dict({"input": inputs, "output": responses})

# Prepare the data
json_file = 'input_response_pairs.json'
data = load_json_dataset(json_file)
dataset = convert_to_dataset(data)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "input",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# Inference with an example

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Aplicatia sa contina 3 ramuri esentiale: 1. Sofer: harta cu traseul sau care sa contina punctele de ridicare si livrare. Timpul estimat pentru ambele . Posibilitatea de a prelua singur o solicitare de la un restaurant sau sa i se acorde livrari. Raport zilnic cu livrarile efectuate.Actualizare in timp real al parcursului sau catre client. 2. Restaurant: preluare comenzi, confirmarea lor si a statusului comenzii in timp real catre client. In orele de varf sa nu mai aiba clientul optiunea de a pune noi comenzi daca dureaza mai mult decat timpul estimat in aplicatie. Solictare sofer si optiunea de a vedea parcursul soferului. 3. Client : sa i-a la cunostinta toate detaliile legate de restaurant ( timp estimate de livrare , timp de preparare etc). sa adauge produse in cos , achizitie cu card sau numerar din aplicatie. Sa primeasca notificari cu fiecare parcurs al comenzii (confirmata, comanda urmeaza sa fie ridicata, comanda a fost ridicata, comanda livrata). Dupa plasarea si confirmarea si preluarea comenzii de catre sofer sa I se prezinte soferul, numarul de contact al acestuia si parcursul sau pe harta. FUNCTIE ADMIN : care poate observa activitatea celor 3 ramuri mentionate si poate efectua un raport pe fiecare. Firma are contracted oar cu restaurante . In aditie mai fac livrari de colete mici sau plicuri la solicitarea clientilor prin urmare ar trebui si o optiune de acest gen in aplicatie unde sa ofere cateva info despre colet (daca e plic sau e o cutie si ce demensiuni/greutate are).", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 3000, use_cache = True)
tokenizer.batch_decode(outputs)