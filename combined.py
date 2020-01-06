"""
File:
Author: Ian Ross
Email: iross@cs.wisc.edu
Github: https://github.com/iross
Description:
    usage:
        doc2vec and word2vec models from a list of docid:
            python combined.py -d docids_50 -v true -w true -o stream_vs_file/dummy -u <xxyyzz> -p <password>
"""

import requests
import json
import gensim, os, sys, fnmatch, shutil
from time import time, strftime, localtime
from datetime import timedelta

import configparser
from requests.auth import HTTPBasicAuth
import argparse
import codecs
import re

from nltk import download
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk.data

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

CONFIG = {}
BASE_URL="http://deepdive2000.chtc.wisc.edu/es/articles/_search"
AUTH=HTTPBasicAuth('', '')
fields=["_id", "contents"]

def format_time(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def normalize_text(text):
    text = text.strip()
    text = text.replace('\n', ' ').replace('\r', '')
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    return norm_text

def parse_data(data):
    if data["hits"]["total"] == 0:
        return
    else:
        for doc in data["hits"]["hits"]:
            xddid = doc["_id"]
            try:
                contents = doc["fields"]["contents"]
                if isinstance(contents, list):
                    contents = contents[0]
                contents = normalize_text(contents)
            except UnicodeDecodeError: continue
#                            print(f"\tYielding {xddid}")
            yield xddid, contents

class SentenceStream(object):
    def __init__(self, docids_name):
        self.docids_name = docids_name
        self.count = 0
    def __iter__(self):
        with open(self.docids_name) as fin:
            # TODO: buffer the query responses like we do for the document stream
            for docid in fin:
                docid = docid.strip()
                # get text from xDD
                query = {'query': {'bool': {'must': {"match_phrase" : {"_id" : docid}} } } , "fields" : fields}
                # TODO: trap failures
                resp = requests.get(BASE_URL, data=json.dumps(query), auth=AUTH)
                data = resp.json()
                if data["hits"]["total"] == 0:
                    continue
                else:
                    xddid = data["hits"]["hits"][0]["_id"]
                    try:
                        contents = data["hits"]["hits"][0]["fields"]["contents"]
                        if isinstance(contents, list):
                            contents = contents[0]
                        sentences = sent_tokenize(contents)
                    except (UnicodeDecodeError, TypeError):
                        continue
                for s in sentences:
                    s = preprocess(s)
                    if len(s) > 0:
                        yield s

class Sentences(object):
    # Read corpus from disk (format: docid\t<contents>)
    # Parse into sentences and stream them back
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
    def __iter__(self):
        with open(self.corpus_file) as fin:
            for line in fin:
                xddid, contents = line.split("\t", 1)
                try:
                    sentences = sent_tokenize(contents)
                except UnicodeDecodeError: continue
                for s in sentences:
                    s = preprocess(s)
                    if len(s) > 0:
                        yield s

class DocumentStream(object):
    def __init__(self, docids_name, buffer_size=200):
        self.docids_name = docids_name
        self.count = sum(1 for line in open(docids_name))
        self.buffer_size=200

    def __iter__(self):
        with open(self.docids_name) as fin:
            docbuffer = []
            processed = 0
            for docid in fin:
                processed+=1
                docid = docid.strip()
                docbuffer.append(docid)
                if len(docbuffer) == self.buffer_size or processed==self.count:
                    # get text from xDD
                    query = {'query': {'ids': {'values': docbuffer } } , "fields" : fields, "size" : self.buffer_size}
                    start_time = time()
                    # TODO: trap failures
                    resp = requests.get(BASE_URL, data=json.dumps(query), auth=AUTH)
#                    print("Got ES response in ", format_time(time() - start_time))
                    data = resp.json()
#                    print(f"Processing {len(data['hits']['hits'])} documents" )
                    if data["hits"]["total"] == 0:
                        continue
                    else:
                        for doc in data["hits"]["hits"]:
                            xddid = doc["_id"]
                            try:
                                contents = doc["fields"]["contents"]
                                if isinstance(contents, list):
                                    contents = contents[0]
                                sentences = sent_tokenize(contents)
                            except UnicodeDecodeError: continue
                            document = preprocess(' '.join(gensim.utils.simple_preprocess(contents)))
#                            print(f"\tYielding {xddid}")
                            yield gensim.models.doc2vec.TaggedDocument(document, [xddid])
                    docbuffer = []

class Documents(object):
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        with open(self.corpus_file) as fin:
            for line in fin:
                xddid, contents = line.split("\t", 1)
                try:
                    sentences = sent_tokenize(contents)
                except UnicodeDecodeError: continue
                document = preprocess(' '.join(gensim.utils.simple_preprocess(contents)))
                yield gensim.models.doc2vec.TaggedDocument(document, [xddid])

stop_words = stopwords.words('english')

def preprocess(line):
    line = word_tokenize(line)  # Split into words.
    line = [w.lower() for w in line]  # Lower the text.
#    line = [w for w in line if not w in stop_words]  # Remove stopwords
#    line = [w for w in line if w.isalpha()] # Remove numbers and punctuation
    return line


### Get word2vec parameters
def parse_config(conf):
    global CONFIG
    CONFIG['vector_size'] = 	conf['word2vecParameters'].getint('vector_size')
    CONFIG['window_size'] = 	conf['word2vecParameters'].getint('window_size')
    CONFIG['min_count'] = 	conf['word2vecParameters'].getint('min_count')
    CONFIG['alpha'] = 	conf['word2vecParameters'].getfloat('alpha')
    CONFIG['min_alpha'] = 	conf['word2vecParameters'].getfloat('min_alpha')
    CONFIG['negative_size'] = conf['word2vecParameters'].getint('negative_size')
    CONFIG['train_epoch'] = 	conf['word2vecParameters'].getint('train_epoch')
    CONFIG['sg'] = 		conf['word2vecParameters'].getint('sg')
    CONFIG['worker_count'] = 	conf['word2vecParameters'].getint('worker_count')
    CONFIG['bigram_phrase_min_count'] = 	conf['bigramPhraseParameters'].getint('min_count')
    CONFIG['bigram_phrase_threshold'] = 	conf['bigramPhraseParameters'].getint('threshold')
    CONFIG['bigram_phrase_progress_per'] = 	conf['bigramPhraseParameters'].getint('progress_per')
    CONFIG['bigram_phrase_delimiter'] = 	conf['bigramPhraseParameters'].getint('delimiter')
    CONFIG['trigram_phrase_min_count'] = 	conf['trigramPhraseParameters'].getint('min_count')
    CONFIG['trigram_phrase_threshold'] = 	conf['trigramPhraseParameters'].getint('threshold')
    CONFIG['trigram_phrase_progress_per'] = 	conf['trigramPhraseParameters'].getint('progress_per')
    CONFIG['trigram_phrase_delimiter'] = 	conf['trigramPhraseParameters'].getint('delimiter')


from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary

def monograms(corpus, output_prefix):
    print("----- Terms -----")
    dct = Dictionary(corpus)
    dct.save(output_prefix + "_dictionary")
    print("Training tf-idf from terms")
    tfidf = gensim.models.TfidfModel([dct.doc2bow(line) for line in corpus], smartirs='ntc')
    tfidf.save(output_prefix + "_tfidf")
    start_time = time()
    print("Training word2vec model with terms")
    print(f"size={CONFIG['vector_size']}, window={CONFIG['window_size']}, \
                                   min_count={CONFIG['min_count']}, workers={CONFIG['worker_count']}, sg={CONFIG['sg']}, \
                                   negative={CONFIG['negative_size']}, alpha={CONFIG['alpha']}, min_alpha = {CONFIG['min_alpha']}, \
                                   iter={CONFIG['train_epoch']}")
    model = gensim.models.Word2Vec(corpus, size=CONFIG['vector_size'], window=CONFIG['window_size'],
                                   min_count=CONFIG['min_count'], workers=CONFIG['worker_count'], sg=CONFIG['sg'],
                                   negative=CONFIG['negative_size'], alpha=CONFIG['alpha'], min_alpha = CONFIG['min_alpha'],
                                   iter=CONFIG['train_epoch'])
    model.save(output_prefix)
    print("Time :", format_time(time() - start_time))
    return model

def bigrams(corpus, output_prefix):
    print("----- Bigram -----")
    if os.path.exists(output_prefix + "_bigram_phrases"):
        bigram_phrases = Phrases.load(output_prefix + "_bigram_phrases")
        print("Loaded bigram phrases")
    else:
        bigram_phrases = Phrases(corpus, min_count=CONFIG["bigram_phrase_min_count"], threshold=CONFIG["bigram_phrase_threshold"], progress_per=CONFIG["bigram_phrase_progress_per"], delimiter=CONFIG["bigram_phrase_delimiter"])
        bigram_phrases.save(output_prefix + "_bigram_phrases")
    bigram_transformer = Phraser(bigram_phrases)

    dct = Dictionary(bigram_transformer[corpus])
    dct.save(output_prefix + "_dictionary_bigram")
    print("Training tf-idf from bigrams")
    bow_corpus = [dct.doc2bow(line) for line in bigram_transformer[corpus]]
    tfidf = gensim.models.TfidfModel(bow_corpus, smartirs='ntc')
    tfidf.save(output_prefix + "_tfidf_bigram")
    print("Training word2vec model with bigrams (may be unnecessary if trigrams work as expected)")
    start_time = time()
    bigram_model = gensim.models.Word2Vec(bigram_transformer[corpus], size=CONFIG['vector_size'], window=CONFIG['window_size'],
                                   min_count=CONFIG['min_count'], workers=CONFIG['worker_count'], sg=CONFIG['sg'],
                                   negative=CONFIG['negative_size'], alpha=CONFIG['alpha'], min_alpha = CONFIG['min_alpha'],
                                   iter=CONFIG['train_epoch'])
    bigram_model.save(output_prefix + "_bigram")
    print("Time :", format_time(time() - start_time))
    return bigram_model

def trigrams(corpus, output_prefix):
    print("----- Trigrams -----")
    if os.path.exists(output_prefix + "_trigram_phrases"):
        trigram_phrases = Phrases.load(output_prefix + "_trigram_phrases")
        print("Loaded trigram phrases")
    else:
        bigram_phrases = Phrases(corpus, min_count=CONFIG["bigram_phrase_min_count"], threshold=CONFIG["bigram_phrase_threshold"], progress_per=CONFIG["bigram_phrase_progress_per"], delimiter=CONFIG["bigram_phrase_delimiter"])
        trigram_phrases = Phrases(bigram_phrases[corpus], min_count=CONFIG["trigram_phrase_min_count"], threshold=CONFIG["trigram_phrase_threshold"], delimiter=CONFIG["trigram_phrase_delimiter"])
        trigram_phrases.save(output_prefix + "_trigram_phrases")
    trigram_transformer = Phraser(trigram_phrases)
    dct = Dictionary(trigram_transformer[corpus])
    dct.save(output_prefix + "_dictionary_trigram")
    print("Training tf-idf from trigrams")
    bow_corpus = [dct.doc2bow(line) for line in trigram_transformer[corpus]]
    tfidf = gensim.models.TfidfModel(bow_corpus, smartirs='ntc')
    tfidf.save(output_prefix + "_tfidf_trigram")
    print("Training word2vec model with trigram")
    start_time = time()
    trigram_model = gensim.models.Word2Vec(trigram_transformer[corpus], size=CONFIG['vector_size'], window=CONFIG['window_size'],
                                   min_count=CONFIG['min_count'], workers=CONFIG['worker_count'], sg=CONFIG['sg'],
                                   negative=CONFIG['negative_size'], alpha=CONFIG['alpha'], min_alpha = CONFIG['min_alpha'],
                                   iter=CONFIG['train_epoch'])
    trigram_model.save(output_prefix + "_trigram")
    print("Time :", format_time(time() - start_time))
    return trigram_model

def doc2vec(corpus, output_prefix):
    # TODO: pass in these args via the .ini file like word2vec
    worker_count=40
    model = gensim.models.doc2vec.Doc2Vec(vector_size=5, min_count=2, epochs=20, workers=worker_count)
    start_time = time()
    print(f"Building vocab.")
    model.build_vocab(corpus)
    print("Vocab built in ", format_time(time() - start_time))

    print(f"Training doc2vec model with {worker_count} workers.")
    start_time = time()
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(output_prefix + "_doc2vec")
    print("Time :", format_time(time() - start_time))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user","-u",type=str,required=True,help="Search endpoint username")
    parser.add_argument("--password","-p",type=str,required=True,help="Search endpoint password")
    parser.add_argument("--docids_name","-d",type=str,required=False,help="File containing the docids of interest.")
    parser.add_argument("--corpus_file","-c",type=str,required=False,help="File containing the contents of interest.")
    parser.add_argument("--output_name","-o",default=None,help="File to save model (default=None)")
    parser.add_argument("--doc2vec", "-v",default=False,help="Do that doc2vec")
    parser.add_argument("--word2vec", "-w",default=False,help="Do that word2vec")
    parser.add_argument("--config", "-f",default="word2vec_default.ini",help="Externally defined config file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    parse_config(config)
    if args.corpus_file is None and args.docids_name is None:
        print("You must specify either the corpus file (-c) or docids file (-d)!")
        sys.exit(1)
    shutil(config, args.output_name + "_config.ini")

    global AUTH
    AUTH=HTTPBasicAuth(args.user, args.password)

    if args.corpus_file is not None:
        print("------ File-based corpus Mode ------")
        sent_corpus = Sentences(args.corpus_file)
        doc_corpus = Documents(args.corpus_file)
        if args.word2vec:
            # temporarily doing individual words + bigrams + trigrams for testing purposes
            model = monograms(sent_corpus, args.output_name)
            bmodel = bigrams(corpus, args.output_name)
            tmodel = trigrams(corpus, args.output_name)
        if args.doc2vec:
            doc2vec(doc_corpus, args.output_name)

    if args.docids_name is not None:
        print("------ Streaming Mode ------")
        streamed_sent_corpus = SentenceStream(args.docids_name)
        streamed_doc_corpus = DocumentStream(args.docids_name)
        if args.word2vec:
            model = monograms(streamed_sent_corpus, args.output_name + "_streamed")
            bmodel = bigrams(streamed_sent_corpus, args.output_name + "_streamed")
            tmodel = trigrams(streamed_sent_corpus, args.output_name + "_streamed")
        if args.doc2vec:
            doc2vec(streamed_doc_corpus, args.output_name + "_streamed")

if __name__ == '__main__':
    main()
