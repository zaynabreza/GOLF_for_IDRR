import pandas as pd

relations = {
    "result": "Contingency.Cause",
    "reason": "Contingency.Cause",
    "succession": "Temporal.Asynchronous",
    "precedence": "Temporal.Asynchronous",
    "synchronous": "Temporal.Synchronous",
    "conjunction": "Expansion.Conjunction",
    "disjunction": "Expansion.Disjunction",
    "similarity": "Comparison.Similarity",
    "contrast": "Comparison.Contrast",
    "arg1-as-denier": "Comparison.Concession",
    "arg2-as-denier": "Comparison.Concession",
    "arg1-as-detail": "Expansion.Level-of-detail",
    "arg2-as-detail": "Expansion.Level-of-detail",
    "arg1-as-instance": "Expansion.Instantiation",
    "arg2-as-instance": "Expansion.Instantiation",
    "arg1-as-goal": "Contingency.Purpose",
    "arg2-as-goal": "Contingency.Purpose",
    "arg1-as-subst": "Expansion.Substitution",
    "arg2-as-subst": "Expansion.Substitution",
    ### MAP to LVL 2
}
def preprocess_csv(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        # Fix common CSV issues, such as unclosed quotes
        line = line.strip().replace('"\t', '"\t').replace('\t"', '\t"')
        # Attempt to balance quotes
        if line.count('"') % 2 != 0:
            line = line.replace('"', '', 1)
        cleaned_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(cleaned_lines))

def convert_to_format(entry):
    arg1 = entry["arg1"]
    arg2 = entry["arg2"]
    gold = entry["majoritylabel_sampled"]
    gold2 = entry["majority_distrlabel20"] # Use this for TRAINING
    conn = entry["domconn_step2"]
    
    mapGold = relations.get(gold, "Unknown Relation")
    mapGold2 = relations.get(gold2, "Unknown Relation")

    
    formatted_entry = " ||| ".join([
        str([mapGold.split('.')[0], mapGold, conn]),
        str([mapGold2.split('.')[0], mapGold2, None]), # Placeholder for gold label 2
        arg1,
        arg2
    ])
    
    return formatted_entry

def convert_csv_to_txt(input_file, output_base):
    df = pd.read_csv(input_file, delimiter='\t', engine='python', on_bad_lines='warn')
    for split_name in ["TRAIN", "DEV", "TEST"]:
        split_df = df[df['split'].str.upper() == split_name]
        output_file = f"{split_name.lower()}.txt"
        
        with open(output_file, 'w') as f:
            for _, row in split_df.iterrows():
                formatted_entry = convert_to_format(row)
                f.write(formatted_entry + "\n")

if __name__ == "__main__":

    output_base_path = '/data'
    input_file = "DiscoGeMcorpus_annotations_WIDE.with_splits.csv"

    preprocessed_file = "preprocessed-DiscoGeMcorpus_annotations_WIDE.with_splits.csv"

    preprocess_csv(input_file, preprocessed_file)


    convert_csv_to_txt(preprocessed_file, output_base_path)