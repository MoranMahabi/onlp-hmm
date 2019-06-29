from solution12 import Submission12


class Submission3(Submission12):

    def train(self, training_treebank_file='data/heb-ctrees.train', percolate=False, parent_encoding=True):
        super().train(training_treebank_file, percolate=False, parent_encoding=True)

    def tree_to_str(self, bp, sentence, start, end, tag):
        if start == end and tag not in bp[start][end]:
            if '@' in tag:
                tag = tag.split('@')[0]
            return f"({tag} {sentence[start - 1]})"
        if len(bp[start][end][tag]) == 2:  # without split, unary chain
            tag, children_tags = bp[start][end][tag]
            chain = self.tree_to_str(bp, sentence, start, end, children_tags[0])
            if '@' in tag:
                tag = tag.split('@')[0]
            return f"({tag} {chain})"

        tag, children_tags, s = bp[start][end][tag]
        left_tag, right_tag = children_tags
        left_str = self.tree_to_str(bp, sentence, start, s, left_tag)
        right_str = self.tree_to_str(bp, sentence, s + 1, end, right_tag)
        if '*' in tag:
            return f"{left_str} {right_str}"
        if '@' in tag:
            tag = tag.split('@')[0]
        return f"({tag} {left_str} {right_str})"
