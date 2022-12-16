import torch
import evaluate
from nltk import wordnet as wn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def modify_snli(model, nonce_choice, stop_words, dataset_dev, tokenizer, idx):
    new_data = []
    for i, row in enumerate(dataset_dev["train"].select(idx)):

        words = list(set([w for w in row["premise"].split(" ") if w not in stop_words]))
        orig_feats = tokenizer(row["premise"], row["hypothesis"], padding=True, truncation=True, return_tensors="pt")
        gd_truth_mapping = ["entailment", "neutral", "contradiction"]
        ground_truth = gd_truth_mapping[row["label"]]
        print("\n--------------------------------\n")
        print("ground truth labels", ground_truth)
        model.eval()
        with torch.no_grad():
            baseline_scores = model(**orig_feats).logits
            baseline_label = [label_mapping[score_max] for score_max in baseline_scores.argmax(dim=1)][0]
            base_score = torch.max(baseline_scores, 1).values[0]

        new_premise = row["premise"]
        new_hypothesis = row["hypothesis"]
        for word in words:
            to_replace = word
            nonce = nonce_choice[i]
            premise = row["premise"].replace(to_replace, nonce)
            hypothesis = row["hypothesis"].replace(to_replace, nonce)
            print(to_replace, nonce)
            nonce_feats = tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt")

            model.eval()
            min_score = base_score

            with torch.no_grad():
                scores = model(**nonce_feats).logits
                label_mapping = ['contradiction', 'entailment', 'neutral']
                label = [label_mapping[score_max] for score_max in scores.argmax(dim=1)][0]
                # print("replaced labels", labels)
                max_score = torch.max(scores, 1).values
                # print(max_score)
            if label != baseline_label:
                print("Original Premise:", row["premise"])
                print("Original Hypothesis:", row["hypothesis"])

                print("Modified Premise:", premise)
                print("Modified Hypothesis:", hypothesis)

                print("Old Label:", baseline_label)
                print("New Label:", label)
                print("Old Score:", base_score)
                print("New Score:", max_score)
                new_premise = premise
                new_hypothesis = hypothesis

                break
            else:
                if max_score < min_score:
                    min_score = max_score
                    new_premise = premise
                    new_hypothesis = hypothesis
                    print("Original Premise:", row["premise"])
                    print("Original Hypothesis:", row["hypothesis"])

                    print("Modified Premise:", premise)
                    print("Modified Hypothesis:", hypothesis)

                    print("Label:", baseline_label)
                    print("Old Score:", base_score)
                    print("New Score:", max_score)
        new_data.append({"premise": new_premise, "hypothesis": new_hypothesis})

    return new_data


def get_answer_squad(model, tokenizer, question, context):
    # for t5
    input_text = "question: %s  context: %s </s>" % (question, context)
    features = tokenizer.batch_encode_plus([input_text], return_tensors='pt')

    out = model.generate(input_ids=features['input_ids'].to(device),
                         attention_mask=features['attention_mask'].to(device))

    return tokenizer.decode(out[0])

def convert_squad_prefix(model, dataset, stop_words, tokenizer, nonces):
    squad_metric = evaluate.load("squad_metric")

    new_rows = []

    for row in dataset:

        question = row["question"]
        context = row["context"]
        references = row["answers"]

        answers = get_answer_squad(model, tokenizer, question, context)

        answers["text"] = answers["text"].replace("<pad>", "").replace("</s>", "")

        base_scores = squad_metric.compute(answers=answers, references = references)

        words = list(set([w for w in row["question"].split(" ") if w not in stop_words]))

        min_f1 = base_scores["f1"]
        min_exact = base_scores["exact_match"]

        final_question = question
        final_context = context
        for word in words:
            to_replace = word
            nonce = np.random.sample(nonces)

            synsets = wn.synsets(word)
            if len(synsets) > 0:
                definition = synsets[0].definition()
            else:
                definition = ""

            new_question = question.replace(to_replace, nonce)
            new_context = context.replace(to_replace, nonce)
            if definition:
                new_context = "{} is defined as: {}".format(nonce, definition) + new_context

            new_answers = get_answer_squad(model, tokenizer, new_question, new_context)
            new_answers["text"] = new_answers["text"].replace("<pad>", "").replace("</s>", "")

            new_references = references
            new_references["text"].replace(to_replace, nonce)

            new_scores = squad_metric.compute(answers=new_answers, references = new_references)

            if new_scores["f1"] < min_f1:
                min_f1 = new_scores["f1"]
                min_exact = new_scores["exact_match"]

                final_question = new_question
                final_context = new_context


        new_rows.append({"question": final_question, "context": final_context})

    return new_rows





