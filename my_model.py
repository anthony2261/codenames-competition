"""
my_model.py: template for models to be used in the TWIMLfest 2020 Codenames competition
Dan Hilgart <dhilgart@gmail.com> and Yuri Shlyakhter <yuri.shlyakhter@gmail.com>
see https://czechgames.com/files/rules/codenames-rules-en.pdf for game rules
"""
"""
------------------------------------------------------------------------------------------------------------------------
                                                    Required Imports
------------------------------------------------------------------------------------------------------------------------
Do not remove these
"""
import TWIML_codenames
import numpy as np

"""
------------------------------------------------------------------------------------------------------------------------
                                                      Your Imports                                                      
------------------------------------------------------------------------------------------------------------------------
Add/modify as necessary
"""
### YOUR CODE HERE
import spacy # after installing, be sure to run 'python -m spacy download en_core_web_lg'
import itertools
import pickle
import json
import os
import time
### END YOUR CODE

"""
------------------------------------------------------------------------------------------------------------------------
                                                  Your Global Variables
------------------------------------------------------------------------------------------------------------------------
Place anything here that you want to be loaded when this module is imported by TWIML_codenames_API_client.py 
For example, if you are loading word vectors, load them here as global variables so they do not have to be loaded each 
    time the generate_clue and generate_guesses functions are called  
"""
### YOUR CODE HERE
nlp = spacy.load("en_core_web_lg") # if OSError: [E050] Can't find model 'en_core_web_lg', run this from command line:
                                   # 'python -m spacy download en_core_web_lg'

# clue_word_distances is a dict containing a 2D numpy array of word distances that have already been pre-calculated in
# order to speed up compute time. The dict also contains 2 additional dicts, both of form {word:index}, for the words on
# the 2 axes: the first axis ('boardwords') is the list of words that the gameboards can be made from, while the second
# axis ('clue_words') is the list of candidate clue words. Boardwords is a list of 400 words, a copy of wordlist.txt.
# clue_words is a list of 6698 nouns based on nounlist.txt sourced from http://www.desiquintans.com/nounlist
# Note: wordlist will be a different set of words for week 2 than week 1. You will need to re-download
# 'clue_word_distances.pkl' from the repo after week 1 is complete before entering your bot in week 2
clue_word_distances = pickle.load(open(os.path.join('distances','clue_word_distances.pkl'),'rb'))

### END YOUR CODE

"""
------------------------------------------------------------------------------------------------------------------------
                                                     Your Functions                                                     
------------------------------------------------------------------------------------------------------------------------
Add/modify functions as necessary
"""
### YOUR CODE HERE
def dist(word1, word2):
    """
    Calculates the vector-distance between two words
    """
    tokens = nlp(word1 + " " + word2)
    return 1 - tokens[0].similarity(tokens[1])

def is_lemma(w1, w2):
    """
    Checks whether or not two words have the same lemma
    """
    return(nlp(str(w1))[0].lemma_ == nlp(str(w2))[0].lemma_)

def list_is_lemma(boardwords, clue):
    """
    Checks if any word on the board and the clue word have the same lemma
    """
    for row in boardwords:
        for word in row:
            if is_lemma(word, clue):
                return True
    return False

def count_guessed_good_words(team_num, boardmarkers):
    """
    Counts the number of correctly guessed words from your team.
    Unused function, written as a tentative to make the guesser smarter.
    """
    count = 0
    for row in boardmarkers:
        for marker in row:
            if marker == team_num:
                count += 1
    return count

def init_json_file(path):
    """
    Creates a JSON file to track each game's clues and guesses (from your team) 
    Unused function, written as a tentative to make the guesser smarter.
    """
    if not os.path.exists(path):
        with open(path, 'w+') as json_file:
            init_json = {"bad_words":[]}
            json.dump(init_json, json_file)

def add_bad_word(path, potential_bad_word):
    """
    Adds potentially incorrect words to the JSON file that is tracking the game.
    Idea: If a word is below the threshold but number of guesses is already reached,
    then this word is not desired <==> bad word.
    Unused function, written as a tentative to make the guesser smarter.
    """
    with open(path) as json_file:
        bad_words = json.load(json_file)["bad_words"]
        if not potential_bad_word in bad_words:
            bad_words.append(potential_bad_word)
    with open(path, 'w') as json_file:
        modified_json = {"bad_words":bad_words}
        json.dump(modified_json, json_file)
    # print("POTENTIAL BAD WORD:" + potential_bad_word)

def remaining_minus_bad(myBestTuplesList, path):
    """
    Returns the number of unguessed words - number of unguessed bad words (saved in the JSON file).
    Unused function, written as a tentative to make the guesser smarter.
    """
    finalList = [tup[0] for tup in myBestTuplesList]
    with open(path) as json_file:
        bad_words = json.load(json_file)["bad_words"]
        for b_word in bad_words:
            if b_word in finalList: finalList.remove(b_word)
    return finalList
### END YOUR CODE

"""
------------------------------------------------------------------------------------------------------------------------
                                                   Required Functions                                                   
------------------------------------------------------------------------------------------------------------------------
These are the two required functions that you must have in your model file.
"""
def generate_clue(game_id, team_num, gameboard: TWIML_codenames.Gameboard):
    """
    This is the function that will be called when your bot is the Spymaster
    Your bot will need to provide a clue_word and a clue_count which will be used by your teammate's bot to guess words
    Make sure to provide a legal clue (see TWIML_codenames.py for how legality is assessed) or the turn will be skipped
    The following inputs will be provided:
    @param game_id (int): the unique identifier for this game. Can be used to locally track info about this game as it 
        plays out
    @param team_num (int): 1 if you are on the first team, 2 if you are on the second team. This matches with the
        gameboard key
    @param gameboard (TWIML_codenames.Gameboard): an object containing the current state of the gameboard. Note that
        this is a copy of the gameboard so any changes made to it will not impact the true gameboard. See
        TWIML_codenames.py for the full details of the TWIML_codenames.Gameboard class. Some useful commands:
            gameboard.boardwords -- 5x5 np.array[str]: the 5x5 grid of words. Remains unchanged after initialization
            gameboard.boardkey -- 5x5 np.array[int]: the key that tells which words belong to which team. Remains
                unchanged after initialization. (1 = team 1, 2 = team 2, 0 = neutral, -1 = assassin)
            gameboard.boardmarkers -- 5x5 np.array[float]: the array that tracks which words have been tapped and what
                was revealed. Starts as an array of np.NaNs. As words are tapped (guessed), the values from the boardkey
                are added for each tapped word.
            gameboard.unguessed_words(team_num[int](optional)) -- list[str]: returns a list of unguessed words for the
                supplied team_num (1 = team 1, 2 = team 2, 0 = neutral, -1 = assassin). If no team_num is supplied, will
                return all remaining unguessed words.
            gameboard.remaining(team_num[int]) -- list[str]: Counts how many cards are left for the supplied team_num

    Please return the outputs as follows:
    @returns clue_word (str): the one-word clue that must not match any part of the remaining words on the board
    @returns clue_count (int): the count of how many board-words are related to the clue word. There are two special
        cases for the clue count:
            A Spymaster can give a clue count of 0 which communicates to the Operative that they should guess words that
                are NOT related to the clue word.
            A spymaster may also give a clue for infinity, allowing the Operative to make as many guesses as they like.
                To give a clue for infinity, provide an int of 10.
    """
    ### YOUR CODE HERE
    threshold = 0.7

    unguessed_good_words = gameboard.unguessed_words(team_num)
    unguessed_bad_words = [word for word in gameboard.unguessed_words() if word not in unguessed_good_words]

    # filter out words that contain, or are contained in, words on the board:
    full_candidates=[]
    
    for candidate in clue_word_distances['clue_words'].keys(): # see definition of clue_word_distances in the 'Your Global Variables' section above.
        duplicate = False
        for unguessed_word in gameboard.unguessed_words():
            if (candidate in unguessed_word) or (unguessed_word in candidate):# or is_lemma(unguessed_word, candidate):
                duplicate = True
                break
        if duplicate == False:
            full_candidates.append(candidate)

    # sample down the list of candidates by a factor of 3 for two reasons: 1) to improve runtime and 2) to avoid getting
    # stuck giving the same clue word over and over again
    candidates = full_candidates
    # candidates = [word for word in
                #   np.random.choice(full_candidates, len(full_candidates)//3, replace=False)]

    # Anthony: a dict of dicts. Key of Inner dict is unguessed_good_word, and inside: key = clue_candidate, pair = distance(of candidate and unguessed word) 
    good_word_distances = {}
    for good_word in unguessed_good_words:
        good_word_distances[good_word] = {}
        for clue_candidate in candidates:
            good_word_distances[good_word][clue_candidate] = clue_word_distances['distances'][clue_word_distances['boardwords'][good_word]][clue_word_distances['clue_words'][clue_candidate]]
        

    # Anthony: a dict of dicts. Key of Inner dict is unguessed_bad_word, and inside: key = clue_candidate, pair = distance(of candidate and unguessed word)
    bad_word_distances = {}
    for bad_word in unguessed_bad_words:
        bad_word_distances[bad_word] = {}
        for clue_candidate in candidates:
            bad_word_distances[bad_word][clue_candidate] = clue_word_distances['distances'][clue_word_distances['boardwords'][bad_word]][clue_word_distances['clue_words'][clue_candidate]]

    clue_count = 0
    clue_word = None
    d = float('Inf')

    for clue_candidate in candidates:
        biggest_candidate_d = 0
        num_safe_words = 0 # THIRD VERSION
        boardwordAndDistance_list = [] # THIRD VERSION
        if(clue_candidate == 'id' or clue_candidate=='wont'):
            continue

        for bad_word in unguessed_bad_words:
            boardwordAndDistance_list.append((bad_word,bad_word_distances[bad_word][clue_candidate]))

        for good_word in unguessed_good_words:
            boardwordAndDistance_list.append((good_word,good_word_distances[good_word][clue_candidate]))

        boardwordAndDistance_list.sort(key=lambda tup: tup[1])
        for word, dist in boardwordAndDistance_list:
            if (word in unguessed_good_words) and (dist < threshold):
                num_safe_words += 1
                biggest_candidate_d = dist # always grows going through the list
                if (dist > threshold):
                    break
            else:
                break

        if num_safe_words > clue_count  and not '-' in clue_candidate and not list_is_lemma(gameboard.boardwords, clue_candidate):#is_lemma(word, clue_candidate):

            clue_count = num_safe_words
            clue_word = clue_candidate
        elif (num_safe_words == clue_count) and (d > biggest_candidate_d) and not '-' in clue_candidate and not list_is_lemma(gameboard.boardwords, clue_candidate):
            clue_word = clue_candidate
            d = biggest_candidate_d
#########
    if not clue_word:
        # if it didn't find a good clue word, return a random word
        clue_word = str(np.random.choice(full_candidates,1)[0])
        clue_count = 1
    
    ### END YOUR CODE
    
    return clue_word, clue_count

def generate_guesses(game_id, team_num, clue_word, clue_count, unguessed_words, boardwords, boardmarkers):
    """
    This is the function that will be called when your bot is the Operative
    Your teammate's bot will provide you with a clue_word and a clue_count. Use them to generate a list of words to
        guess.
    The following inputs will be provided:
    @param game_id (int): the unique identifier for this game. Can be used to locally track info about this game as it 
        plays out
    @param team_num (int): 1 if you are on the first team, 2 if you are on the second team. This matches with the
        boardmarkers array
    @param clue_word (str): the one-word clue from your spymaster
    @param clue_count (int): the count of how many board-words are related to the clue word. There are two special
        cases for the clue count:
            A Spymaster can give a clue count of 0 which communicates to the Operative that they should guess words that
                are NOT related to the clue word.
            A spymaster may also give a clue for infinity, allowing the Operative to make as many guesses as they like.
                An int of 10 is used to represent a clue for infinity.
    @param unguessed_words (list[str]): a 1-d list of all the remaining words that have not yet been tapped
    @param boardwords (5x5 np.array[str]): the 5x5 grid of words. Remains unchanged after initialization
    @param boardmarkers (5x5 np.array[float]): the array that tracks which words have been tapped and what was revealed.
        Starts as an array of np.NaNs. As words are tapped (guessed), the team number (1 = team 1, 2 = team 2,
        0 = neutral, -1 = assassin) of each tapped word are added.

    Please return the outputs as follows:
    @returns guesses (list[str]): a list of the words that you would like to tap in the order you want them tapped.
        Words on the list will continue to be tapped until a word is tapped that is not one of your team's words
    """
    ### YOUR CODE HERE
    # Algorithm based on the following paper:
    # Cooperation and Codenames:Understanding Natural Language Processing via Codenames
    # by A. Kim, M. Ruzmaykin, A. Truong, and A. Summerville 2019
    threshold_for_guessing = 0.83 # Anthony: was 0.7

    # path = os.path.join('guessing',f'guess_{game_id}_{team_num}.json')
    # init_json_file(path)

    # myBestTuplesList = []
    # myGuesses = []

    # # Generate list of (word, distance) tuples "myBestTuplesList"
    # for word in unguessed_words:
    #     if clue_word in clue_word_distances['clue_words'].keys():
    #         curr_dist = clue_word_distances['distances'][clue_word_distances['boardwords'][word], clue_word_distances['clue_words'][clue_word]]
    #     else:
    #         curr_dist = dist(clue_word, word)
    #     myBestTuplesList.append((word, curr_dist))
    # # Sort myBestTuplesList by increasing value of distance
    # myBestTuplesList.sort(key=lambda tup: tup[1])

    # # Pick top 'clue_count' words (and remove them from myBestTuplesList)
    # while len(myGuesses) < clue_count:
    #     best_tuple = myBestTuplesList[0]
    #     if best_tuple[1] < threshold_for_guessing:
    #         myGuesses.append(best_tuple[0])
    #         myBestTuplesList.remove(best_tuple)

    # # If we picked 'clue_count' words and the first remaining tuple inside myBestTuplesList has distance < 0.7, then it's a bad word
    # if myBestTuplesList[0][1] < 0.7:
    #     potential_bad_word = myBestTuplesList[0][0]
    #     add_bad_word(path, potential_bad_word) # add bad word to json in case it wasn't added

    # # get list of remaining words without the bad words
    # remaining_ambiguous_words = remaining_minus_bad(myBestTuplesList, path)

    # # Get how many good words left (excluding current guesses)
    # total_good_words = 8 + (team_num % 2)
    # good_words_left = total_good_words - count_guessed_good_words(team_num, boardmarkers) - len(myGuesses) # my guesses are impeccable

    # # if what's left is unguessed_good and unguessed_known_bad, then add the unguessed_good
    # if good_words_left == len(remaining_ambiguous_words):
    #     # print(f"It's alive!!! guesses before: {myGuesses}")
    #     myGuesses += remaining_ambiguous_words
    #     # print(f"Guesses after: {myGuesses}")

    # if len(myGuesses) == 0:
    #     myGuesses.append(str(np.random.choice(unguessed_words,1)[0]))
    # return myGuesses

    guesses = []

    if clue_word in clue_word_distances['clue_words'].keys():
        while len(guesses) < clue_count:
            best = None
            d = float('Inf')
            for word in unguessed_words:
                # distance = clue_word_distances['distances'][clue_word_distances['boardwords'][word], clue_word_distances['clue_words'][clue_word]]
                distance = clue_word_distances['distances'][clue_word_distances['boardwords'][word]][clue_word_distances['clue_words'][clue_word]]
                if (distance < d):
                    d = distance
                    best = word
            if (best and d < threshold_for_guessing):
                guesses.append(best)
                unguessed_words.remove(best)
            else:
                break

    else:
        while len(guesses) < clue_count:
            best = None
            d = float('Inf')
            for word in unguessed_words:
                distance = dist(clue_word, word)
                if (distance < d):
                    d = distance
                    best = word
            if (best and d < threshold_for_guessing):
                guesses.append(best)
                unguessed_words.remove(best)
            else:
                break

    if len(guesses) == 0:
        guesses.append(str(np.random.choice(unguessed_words,1)[0]))
    ### END YOUR CODE
    
    return guesses
