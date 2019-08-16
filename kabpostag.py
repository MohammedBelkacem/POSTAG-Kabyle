import nltk
import sys

# tafelwit igebren tiyuga awal/tacreḍt n wawal
kab_tags_words1 = [ ]

# aḍris n ulmad
text=""
#izirig deg uḍris
ligne=""
first=0
# asali n uḍris n ulmad i ulguritm
for ligne in open("c:/tal/corpuspos.txt",encoding='utf-8'):
    if (first!=0):
     kab_tags_words1.append( ("START", "START") )
     #ligne=ligne.lower()
     ligne=ligne.replace("\n","")
     a=ligne.split(" ")

     for i in a:
        b=i.split("/")
        try:
         kab_tags_words1.append( (b[1],b[0]))
        except:
            print (a)
            exit ()
     kab_tags_words1.append( ("END", "END") )
    first=1

      #split a couple



# caclcul de la distribution des probabilités conditionnelles de la succession des étiquettes (classes grammaticales en kabyle)

cfd_tagwords = nltk.ConditionalFreqDist(kab_tags_words1)

cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

# anadi n ticraḍ yellan deg uḍris

kab_tags = [tag for (tag, word) in kab_tags_words1 ]

# asiḍen n tseqqaṛ n umseḍfer n ticraḍ
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(kab_tags))

cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)


distinct_tags = set(kab_tags)

# anadi n ticraḍ n tefyir ur yettwacerḍen ara

tafyirt = "ttnadiɣ ad issineɣ ."
sentence = tafyirt.split()
sentlen = len(sentence)

viterbi = [ ]

backpointer = [ ]

first_viterbi = { }
first_backpointer = { }
for tag in distinct_tags:
    # don't record anything for the START tag
    if tag == "START": continue
    first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
    first_backpointer[ tag ] = "START"


viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
print( "Awal", "'" + sentence[0] + "'", "Agzum n umseḍfaṛ n 2 n ticraḍ ifazen:", first_backpointer[ currbest], currbest)
# print( "Word", "'" + sentence[0] + "'", "current best tag:", currbest)

for wordindex in range(1, len(sentence)):
    this_viterbi = { }
    this_backpointer = { }
    prev_viterbi = viterbi[-1]

    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue

        # if this tag is X and the current word is w, then
        # find the previous tag Y such that
        # the best tag sequence that ends in X
        # actually ends in Y X
        # that is, the Y that maximizes
        # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        # The following command has the same notation
        # that you saw in the sorted() command.
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: \
            prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

        # Instead, we can also use the following longer code:
        # best_previous = None
        # best_prob = 0.0
        # for prevtag in distinct_tags:
        #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
        #    if prob > best_prob:
        #        best_previous= prevtag
        #        best_prob = prob
        #
        this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
            cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
        this_backpointer[ tag ] = best_previous

    currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
    print( "Awal ", "'" + sentence[ wordindex] + "'", "Agzum ifazen iwatan n 2 n ticraḍ:", this_backpointer[ currbest], currbest)
    # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)


    # done with all tags in this iteration
    # so store the current viterbi step
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)


# done with all words in the sentence.
# now find the probability of each tag
# to have "END" as the next tag,
# and use that to find the overall best sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# best tagsequence: we store this in reverse for now, will invert later
best_tagsequence = [ "END", best_previous ]
# invert the list of backpointers
backpointer.reverse()

# go backwards through the list of backpointers
# (or in this case forward, because we have inverter the backpointer list)
# in each case:
# the following best tag is the one listed under
# the backpointer for the current best tag
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()
print( "Tafyirt d:", end = " ")
for w in sentence: print( w, end = " ")
print("\n")
print( "Agzum n ticraḍ ifazen :", end = " ")
for t in best_tagsequence: print (t, end = " ")
print("\n")
print( "Tiseqqaṛ n ugzum ifazen d:", prob_tagsequence*(10**55))
