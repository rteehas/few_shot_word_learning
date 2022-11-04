

def get_nonces(fname):
    lines = []
    with open(fname, 'r', encoding='latin-1') as f:
        # correct = {line[:3]: line for line in f}
        for line in f:
            lines.append(line)
    nonce_toks = []
    for line in lines:
        tok = "<{}_nonce>".format(line[1].lower())
        nonce_toks.append(tok)

    return nonce_toks

def get_basic_dataset(src_name, tgt_fname):
    lines = []
    with open(src_name, 'r', encoding='latin-1') as f:
        # correct = {line[:3]: line for line in f}
        for line in f:
            lines.append(line)

    tgt_lines = []
    with open(tgt_fname, 'r', encoding='latin-1') as f:
        # correct = {line[:3]: line for line in f}
        for line in f:
            tgt_lines.append(line)

    tgt_lines = [l.strip("\n").split("\t") for l in tgt_lines]
    lines = [l.strip("\n").split("\t") for l in lines]

    dataset_dict = {}
    for l in lines[1:]:
        if l[0]:
            trial = l[0].split("_")
            # print(trial)
            # if trial[0] == "100":
            #   print(trial[0], trial[1].lower())
            if trial[0] not in dataset_dict:
                dataset_dict[trial[0]] = {}
            if trial[1].lower() not in dataset_dict[trial[0]]:
                dataset_dict[trial[0]][trial[1].lower()] = {}
            dataset_dict[trial[0]][trial[1].lower()][trial[2]] = l[1]

    return dataset_dict
