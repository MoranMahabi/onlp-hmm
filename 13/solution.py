import dill

from pcfg import PCFG
from spec import Spec
from util import transliteration

DEBUG = False


class Submission(Spec):
    pcfg = None

    def train(self, training_treebank_file='data/heb-ctrees.mini', percolate=True):
        pickle_file = "./pcfg-p.p" if percolate else "./pcfg.p"
        if DEBUG:
            with open(pickle_file, "rb") as f:
                self.pcfg = dill.load(f)
            return

        self.pcfg = PCFG(training_treebank_file)

        # self.pcfg.smooth_unknowns()
        

        self.pcfg.binarize()

        if percolate:
            self.pcfg.percolate()

            self.pcfg.validate()
        
        self.pcfg.add_delta_smoothing()

        self.pcfg.reverse()

        with open(pickle_file, "wb") as f:
            dill.dump(self.pcfg, f)

    def parse(self, sentence):
        raise NotImplementedError()

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        with open(output_treebank_file, 'w') as f:
            for i, sentence in enumerate(sentences):
                print(f"Parsing {i + 1}/{len(sentences)}...")
                f.write(self.parse(sentence))
                f.write('\n')
                f.flush()

    @staticmethod
    def to_heb(sentence):
        " ".join([''.join([transliteration.to_heb(c) for c in w]) for w in sentence if not w.startswith('yy')])
