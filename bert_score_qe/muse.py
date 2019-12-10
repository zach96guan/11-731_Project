import io
import numpy as np
import re

"""class MuseTokenizer:
    def __init__(self, emb_path, nmax=50000): 
        self.embeddings, self.id2word, self.word2id = load_vec(src_path, nmax)
        self.pad_token_id = 0 """

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) % 10000 == 0:
                print(len(word2id))
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def read_corpus(file_path, word2id, embeddings):
    sentences = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        vectors_sent = []
        for word in sent:
            word = word.lower()
            """if len(word) == 0:
                continue
            if word[0] == '\"':
                word = word[1:]
            if len(word) == 0:
                continue
            if word[-1] == '.':
                word = word[:-1]"""
            word = re.sub('[\",.:!?()]', '', word)
            if "-" in word:
                subwords = word.split('-')
            else:
                subwords = [word]
            for subword in subwords:
                if subword not in word2id:
                    #print(word)
                    continue
                index = word2id[subword]
                vect = embeddings[index]
                vectors_sent.append(vect)
        """if len(sentences) == 1:
            print(len(vectors_sent))"""
        sentences.append(vectors_sent)
    return sentences

def main():
    src_vec_path = 'E:/Projects/wiki.multi.de.vec'
    tgt_vec_path = 'E:/Projects/wiki.multi.en.vec'
    nmax = 10000  # maximum number of word embeddings to load

    src_embeddings, src_id2word, src_word2id = load_vec(src_vec_path, nmax)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_vec_path, nmax)
    #print(src_word2id)

    src_path = './newstest2019-deen-src.de'
    tgt_path = './de-en/newstest2019.Facebook_FAIR.6750.de-en'
    src_sentences = read_corpus(src_path, src_word2id, src_embeddings)
    tgt_sentences = read_corpus(tgt_path, tgt_word2id, tgt_embeddings)
    
    scores = []
    for i in range(len(src_sentences)):
        src_sent = src_sentences[i]
        tgt_sent = tgt_sentences[i]
        if len(src_sent) == 0 or len(tgt_sent) == 0:
            #print(i)
            scores.append(1)
            continue

        # compute recall
        cur_sum = 0
        for src_wrd in src_sent:
            cur_max = -2
            for tgt_wrd in tgt_sent:
                cur = np.dot(src_wrd, tgt_wrd)
                if cur > cur_max:
                    cur_max = cur
            cur_sum += cur_max
        R = cur_sum / len(src_sent)
        #print(R)

        # compute precision
        cur_sum = 0
        for tgt_wrd in tgt_sent:
            cur_max = -2
            for src_wrd in src_sent:
                cur = np.dot(src_wrd, tgt_wrd)
                if cur > cur_max:
                    cur_max = cur
            cur_sum += cur_max
        P = cur_sum / len(tgt_sent)

        # compute F1
        F = 2 * P * R / (P + R)
        #print("%.5f, %.5f %.5f" % (R, P, F))
        scores.append(F)

if __name__ == '__main__':
    main()