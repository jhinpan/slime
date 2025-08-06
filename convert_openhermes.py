from datasets import load_dataset

print("Loading OpenHermes-2.5 dataset from HuggingFace...")
ds = load_dataset("teknium/OpenHermes-2.5")["train"]
print(f"Dataset loaded. Total samples: {len(ds)}")

def convert(sample):
    conversations = sample["conversations"]

    def convert_role(role):
        if role == "human":
            return "user"
        elif role == "gpt":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["from"]),
            "content": turn["value"],
        }
        for turn in conversations
    ]

    return {"messages": messages}

print("Converting dataset to Slime format...")
ds = ds.map(convert)

print("Saving to /root/openhermes2_5.parquet...")
ds.to_parquet("/root/openhermes2_5.parquet")
print("Dataset saved successfully!")