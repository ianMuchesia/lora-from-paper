def format_prompt(data_row):
    return (
        f"### Instruction:\n"
        f"{data_row['instruction']}\n\n"
        f"### Response:\n"
        f"{data_row['output']}"
    )
