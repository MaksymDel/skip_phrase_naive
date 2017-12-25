import sys
import os.path

def build_dataset_skip_gram(path_in, path_out, window_size=2):
    with open(path_in, 'r') as fin, open(path_out, 'a+') as fout:
        print("Start processing file `%s`..." % path_in)
        lines = fin.readlines()
        result_lines = []       
        for line in lines:
            examples = process_line_skip_gram(line)
            examples = [" ".join(ex) for ex in examples]
            result_lines.extend(examples)

        fout.writelines("\n".join(result_lines))
        print("Skip-gram dataset is wrote to the `%s`." % path_out)

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


def build_dataset_skip_phrase(path_data):
    with open(path_data, 'r') as f:
        lines = f.readlines()

    return lines

def main(argv):
    """ 
    Builds skip-gram or skip-phrase dataset from dataset that contains sentences
    one sentence per line.

    Assumes input dataset is already tokenized and truecased
   
    Usage:
    python build dataset.py type path_in, path_out, window_size=2  
    
    Parameters
    ----------
    type: `gram` or `phrase`, depending on what do you want, usual skip-gram with words, 
        or skip-phrase that embeds phrases
    path_in: path to input dataset of lines
    path_out: path to result dataset ready to be passed to model
    window_size: number of context words to use on one side 
    """
    mtype = argv[0]
    pin = argv[1]
    pout = argv[2]
    window_size = argv[3]

    if os.path.isfile(pout):
        raise OSError("Output file alredy exist, please delete it or specify another output filename. Terminatig...")

    if mtype == 'gram':
        build_dataset_skip_gram(pin, pout, window_size)
    elif mtype == 'phrase':
        build_dataset_skip_gram(pin, pout, window_size)

if __name__ == "__main__":
    main(sys.argv[1:])


"""
#
Pseudocode to handle cases where dataset does not fit into a memory
#
def build_dataset_skip_gram(path_in, path_out, window_size=2):
    with open(path_in, 'r') as fin, open(path_out, 'a') as fout:
        print("Start processing file %s...", path_in)
        line = fin.readline()
        while line:
            examples = process_line_skip_gram(line)
            for ex in examples:
                fout.writelines(" ".join(ex))
                fout.write('\n')
        print("Skip-gram dataset is wrote to the %s.", path_out)
"""

