import os.path as path
import time
from os import remove

from util.tree.builders import list_tree_from_sequence
from util.tree.get_yield import get_yield


def drive(parser_class_under_test, output_treebank_file='output/predicted.txt'):
    testing_treebank_file = 'data/heb-ctrees.gold'

    parser = parser_class_under_test()

    # invoke the training
    before = time.time()
    parser.train()
    print(f'training took {time.time() - before:.1f} seconds')

    if path.exists(output_treebank_file):
        remove(output_treebank_file)

    sentences = []
    with open(testing_treebank_file, 'r') as test_set:
        for bracketed_notation_tree in test_set:
            list_tree = list_tree_from_sequence(bracketed_notation_tree)
            tree_yield = get_yield(list_tree)
            sentences.append(tree_yield)

    # parse
    before = time.time()
    parser.write_parse(sentences, output_treebank_file)
    print(f'parsing took {time.time() - before:.1f} seconds')

    # you can use other output paths for your experiments,
    # but for the final submission, you must to use the
    # default one used here:                    
    assert path.exists(output_treebank_file), 'your write_parse method did not write its output!'

    print('thanks for the parsing!\n')
