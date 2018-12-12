
import gzip
import gensim
import logging
import os
import sys  
import io


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

#with dlya raboti s neupravlyami resursami
#enumerate avtomaticheski shet4ik prostavlyat zna4enie v strokah listah ili massivah
def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            #print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 1000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../wiki.sakha.txt.gz")
    #print(abspath)
    #text = io.open(abspath + '/wiki.sakha.txt', 'r', encoding='utf-8').read()

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    #print(documents)
    #sys.exit(1)
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    # save only the word vectors
    model.wv.save(os.path.join(abspath, "../vectors/defaultsakha"))

    # look up top 6 words similar to 'polite'
    w1 = ["кыыл"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))
    w1 = ["суруйаачыта"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))
    w1 = ["куорат"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))
    w1 = ["маҥан"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))    
    w1 = ["сырдык"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))    
    w1 = ["сүүрэр"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))    



