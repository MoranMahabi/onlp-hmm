from pcfg import PCFG
from spec import Spec

SKIP_TRAIN = False
DEBUG = True

if DEBUG:
    import dill


class Submission(Spec):
    pcfg = None

    def train(self, training_treebank_file='data/heb-ctrees.train', percolate=True, parent_encoding=False):
        pickle_file = "./pcfg-p.pe" if parent_encoding else ("./pcfg-p.p" if percolate else "./pcfg.p")
        if SKIP_TRAIN:
            with open(pickle_file, "rb") as f:
                self.pcfg = dill.load(f)
            return

        self.pcfg = PCFG(training_treebank_file, parent_encoding)

        self.pcfg.normalize_and_smooth()

        self.pcfg.binarize()

        if percolate:
            self.pcfg.percolate()

        if DEBUG:
            self.pcfg.validate()

        self.pcfg.reverse()

        if DEBUG:
            with open(pickle_file, "wb") as f:
                dill.dump(self.pcfg, f)

    def parse(self, sentence):  # to be overridden by derived classes
        raise NotImplementedError()

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        with open(output_treebank_file, 'w') as f:
            for i, sentence in enumerate(sentences):
                print(f"Parsing {i + 1}/{len(sentences)}...")
                f.write(self.parse(sentence))
                f.write('\n')
                f.flush()
