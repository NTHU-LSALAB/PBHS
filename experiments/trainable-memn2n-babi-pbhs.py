from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.pbhs import PBHS
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import ray

from ray import tune
from ray.tune.trial import Resources
from ray.tune.examples.utils import set_keras_threads
from babi_helpers import get_and_parse_babi_data
from ray.tune.resource_ray_executor import ResourceExecutor
from hypersched.tune import ResourceTrainable
import tarfile
import re
import random
import argparse
from functools import reduce
parser = argparse.ArgumentParser(description="PyTorch MemN2N Example")
parser.add_argument(
    "--cpus", type=int, help="the number of CPUs")
parser.add_argument(
    "--deadline", type=int, help="experiment time limit")
DEFAULT_CONFIG = {
    "lstm_size": 128,
    "embedding": 64,
    "dropout": 0.3,
    "opt": "rmsprop",
    "challenge_type": "single_supporting_fact_10k",
    "threads": 4,
    "batch_size": 32,
}

DEFAULT_HSPACE = {
    "lstm_size": tune.uniform(4, 128),
    "embedding": tune.uniform(32, 128),
    "dropout": tune.uniform(0.1, 0.5),
    "opt": tune.choice(["rmsprop", "adam", "sgd"]),
    # "lstm_size": 32,
    # "embedding": 64,
    # "dropout": 0.3,
    # "opt": "rmsprop",
}

# this just loads the file
def tokenize(sent):
    """Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    res = re.findall(r"[\w']+|[.,!?;]", sent)
    # res = [x.strip() for x in re.split('(\W+)?', sent) if x and x.strip()]

    return res


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))
try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise


challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_'
                                  'single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_'
                                'two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
with tarfile.open(path) as tar:
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')
class BABITrainable(tune.Trainable):

    metric = "mean_accuracy"

    def setup(self, config):
        # placeholders
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                    output_dim=int(config["embedding"])))
        input_encoder_m.add(Dropout(config["dropout"]))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                    output_dim=query_maxlen))
        input_encoder_c.add(Dropout(config["dropout"]))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size,
                                    output_dim=int(config["embedding"]),
                                    input_length=query_maxlen))
        question_encoder.add(Dropout(config["dropout"]))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(int(config["lstm_size"]))(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(config["dropout"])(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer=config["opt"], loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        self.model = model

    def step(self):
        result = self.model.fit([inputs_train, queries_train], answers_train,
            batch_size=32,
            epochs=1,
            validation_data=([inputs_test, queries_test], answers_test))
        print("keras result", result.history)
        res = {k: v[-1] for k, v in result.history.items()}
        
        res["mean_accuracy"] = res["val_accuracy"]
        # res.update(samples=total_samples)
        return res

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.model
        self.model = load_model(path)

    @classmethod
    def name_creator(cls, trial):
        name = f"{trial.trainable_name}"
        params = "_".join(
            [
                f"{k}={trial.config[k]:3.3f}"
                for k in ["lstm_size", "embedding", "dropout"]
            ]
        )
        params += f"{trial.config['opt']}"
        return name + params

    @classmethod
    def to_atoms(cls, resources):
        return int(resources.cpu)

    @classmethod
    def to_resources(cls, atoms):
        return Resources(cpu=atoms, gpu=0)

if __name__ == "__main__":
    args = parser.parse_args()
    # ray.init(address=args.ray_address, num_cpus=16, num_gpus=4)
    cpus = args.cpus
    ray.init(num_cpus=cpus, object_store_memory=2 * (10**10))
    max_iters = 300
    experiment_deadline = args.deadline

    hs = FIFOScheduler()
    ahb = AsyncHyperBandScheduler(grace_period=20,max_t=max_iters)
    pbhs = PBHS(gpus_limit=cpus, metric_key="mean_accuracy", time_limit=experiment_deadline, max_iteration=max_iters, exploration_ratio=1.0, dynamic_exploration = True, check_n_prediction_records=2)
    analysis = tune.run(
        BABITrainable,
        metric="mean_accuracy",
        mode="max",
        scheduler=pbhs,
        # trial_executor=ResourceExecutor(deadline_s=experiment_deadline),
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": max_iters,
        },
        # resources_per_trial=BABITrainable.to_resources(1)._asdict(),
        resources_per_trial={
            "cpu": 1,
            "gpu": 0,
        },
        num_samples=2000,
        checkpoint_at_end=False,
        checkpoint_freq=0,
        config=DEFAULT_HSPACE)

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))