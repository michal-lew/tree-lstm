from collections import defaultdict


def parse():
    word2idx = {'UNK': 0}

    train_paths = dict()
    test_paths = dict()

    train_paths['labels'] = 'training-treebank/rev_labels.txt'
    train_paths['parents'] = 'training-treebank/rev_parents.txt'
    train_paths['rels'] = 'training-treebank/rev_rels.txt'
    train_paths['sentence'] = 'training-treebank/rev_sentence.txt'

    test_paths['labels'] = 'poleval_test/polevaltest_labels.txt'
    test_paths['parents'] = 'poleval_test/polevaltest_parents.txt'
    test_paths['rels'] = 'poleval_test/polevaltest_rels.txt'
    test_paths['sentence'] = 'poleval_test/polevaltest_sentence.txt'

    def md(paths, train=False):
        data = dict()

        for t in paths:
            data[t] = [line.rstrip().lower() for line in open(paths[t]).readlines()]

        trees = []

        for tree in data['parents']:
            trees.append(defaultdict(list))
            t = tree.split()
            for i in range(len(t)):
                word_nb = i
                parent_nb = int(t[i]) - 1
                trees[-1][parent_nb].append(word_nb)

        labels = [[int(j) for j in i.split()] for i in data['labels']]
        labels = [[2 if j == -1 else j for j in i] for i in labels]
        sentences = [i.split() for i in data['sentence']]

        # lemmatize

        if train:
            for sentence in sentences:
                for word in sentence:
                    if word not in word2idx:
                        idx = len(word2idx)
                        word2idx[word] = idx
            sentences = [[word2idx[word] for word in sentence] for sentence in sentences]
        else:
            sentences = [[word2idx[word] if word in word2idx else word2idx['UNK'] for word in sentence] for sentence in sentences]

        rels = [i.split() for i in data['rels']]

        return trees, labels, sentences, rels

    train_data = zip(*md(train_paths, train=True))
    test_data = zip(*md(test_paths))

    return word2idx, train_data, test_data


def left_traverse(node, tree, l):
    for i in range(len((tree[node]))):
        left_traverse(tree[node][i], tree, l)
    l.append(node)
