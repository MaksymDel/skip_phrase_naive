def build_dataset_skip_gram(path_to_in, path_to_out, window_size=2):
    with open(path_to_in, 'r') as fin, open(path_to_out, 'a+') as fout:
        print("Start processing file %s...", path_to_in)
        lines = fin.readlines()
        result_lines = []       
        for line in lines:
            examples = process_line_skip_gram(line)
            examples = [" ".join(ex) for ex in examples]
            result_lines.extend(examples)

        print(result_lines)

        fout.writelines("\n".join(result_lines))
        print("Skip-gram dataset is wrote to the %s.", path_to_out)

def process_line_skip_gram(line, window_size=2):
    words = line.split()

    if len(words) < 2:
        return None
    
    skip_gram_examples = []
    for i in range(len(words)):
        try:
            for j in reversed(range(window_size)):
                j += 1 # to make window index start at 1
                if (i - j) < 0:
                    continue
                skip_gram_examples.append((words[i], words[i-j]))  

            for j in range(window_size):
                j += 1 # to make window index start at 1
                skip_gram_examples.append((words[i], words[i+j]))
        except IndexError:
            pass

    return skip_gram_examples


def build_dataset_skip_phrase(path_to_data):
    with open(path_to_data, 'r') as f:
        lines = f.readlines()

    return lines

"""
def build_dataset_skip_gram(path_to_in, path_to_out, window_size=2):
    with open(path_to_in, 'r') as fin, open(path_to_out, 'a') as fout:
        print("Start processing file %s...", path_to_in)
        line = fin.readline()
        while line:
            examples = process_line_skip_gram(line)
            for ex in examples:
                fout.writelines(" ".join(ex))
                fout.write('\n')
        print("Skip-gram dataset is wrote to the %s.", path_to_out)
"""