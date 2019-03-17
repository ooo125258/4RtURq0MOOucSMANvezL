from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os
import pickle


def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm.
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
    AM = {}

    # Read training data
    training_data = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
    # initialize P(f | e)
    AM = initialize(training_data["eng"], training_data["fre"])

    # Iterate between E and M steps
    # for a number of iterations:
    temp_AM = AM
    for i in range(max_iter):
        temp_AM = em_step(AM, training_data["eng"], training_data["fre"])
    temp_AM["SENTSTART"] = {}
    temp_AM["SENTSTART"]["SENTSTART"] = 1
    temp_AM["SENTEND"] = {}
    temp_AM["SENTEND"]["SENTEND"] = 1
    AM = temp_AM

    save_file = open(fn_AM + '.pickle', 'wb')
    pickle.dump(AM, save_file)
    save_file.close()

    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider

	Return:
	    training data: {'eng':[], 'fre':[]}
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
    # TODO
    counter = 0
    training_set = {'eng': [], 'fre': []}
    for root, dirs, files in os.walk(train_dir, topdown=False):
        for file in files:
            if not (len(file) > 2 and file[-1] == 'e' and file[-2] == '.'):  # .e
                continue

            e_fullName = os.path.join(train_dir, file)
            f_fullName = e_fullName[:-1] + 'f'
            if not os.path.exists(f_fullName):
                # To remove eng without fre
                continue
            e_file = open(e_fullName)
            f_file = open(f_fullName)

            e_readLine = e_file.readline()
            f_readLine = f_file.readline()

            while e_readLine:  # "" is false directly
                if not f_readLine:
                    continue
                training_set['eng'].append(preprocess(e_readLine, 'e'))
                training_set['fre'].append(preprocess(f_readLine, 'f'))
                counter += 1

                if counter >= num_sentences:
                    # The time is now
                    e_file.close()
                    f_file.close()
                    return training_set
                e_readLine = e_file.readline()
                f_readLine = f_file.readline()
            e_file.close()
            f_file.close()
    return training_set


def initialize(eng, fre):
    """
    list: eng
    list: fre
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    # TODO
    # check inbalance - although it's impossible
    if len(eng) != len(fre):
        print("Function initialize: \
        unbalanced eng and fre len: {} and {}".format(len(eng), len(fre)))
        return {}
    counting = {}
    AM = {}

    for i in range(len(eng)):
        eng_words = eng[i].split()
        fre_words = fre[i].split()

        # Count all relationship from each english word to each french word!
        for j in range(len(eng_words)):
            if j not in counting:
                counting[j] = {}
            for k in range(len(fre_words)):
                # There is a relation for this english word and all french word in the selected sentence
                if k not in counting[j]:
                    counting[j][k] = 1
                else:
                    counting[j][k] += 1

    for eng_token in counting:
        AM[eng_token] = {}
        length_fre_tokens_for_this_eng_token = len(counting[eng_token])
        for fre_token in counting:
            if length_fre_tokens_for_this_eng_token == 0:
                # Although I don't think it can be zero, if there is a loop
                AM[eng_token][fre_token] = 0
            else:
                AM[eng_token][fre_token] = 1.0 / length_fre_tokens_for_this_eng_token
        return AM


def em_step(AM, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    # TODO

    tcount = {}
    total = {}

    for e in AM:
        # set total(e) to 0 for all e
        total[e] = 0
        tcount[e] = {}
        for f in AM[e]:
            # set tcount(f, e) to 0 for all f, e
            tcount[e][f] = 0

    # for each sentence pair (F, E) in training corpus:
    for pair_idx in len(eng):
        # for each unique word f in F:
        f_unique_words = unique_word(fre[pair_idx])
        for word_f in f_unique_words:
            denom_c = 0
            e_unique_words = unique_word(eng[pair_idx])
            # for each unique word e in E:
            for word_e in e_unique_words:
                # denom_c += P(f|e) * F.count(f)
                denom_c += AM[e][f] * f_unique_words[f]
            # for each unique word e in E:
            for word_e in e_unique_words:
                value_added = AM[e][f] * f_unique_words[f] * e_unique_words[e] / denom_c
                # tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                tcount[e][f] += value_added
                # total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                total[e] += value_added
    # for each e in domain(total(:)):
    for e in total:
        # for each f in domain(tcount(:,e)):
        for f in tcount[e]:
            # P(f|e) = tcount(f, e) / total(e)
            AM[e][f] = tcount[e][f] / total[e]
    return AM


def unique_word(sentence):
    tokens = sentence.split()
    # remove duplicate by sets!
    # return list(set(tokens))

    # return a dict. The unique words are token and the times of appearance is value
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-12.php
    counts = {}
    for word in tokens:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def unique_only(sentence):
    tokens = sentence.split()
    return list(set(tokens))