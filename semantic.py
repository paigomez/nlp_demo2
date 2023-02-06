import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
# cat and monkey: 0.5929929675536907 

print(word3.similarity(word2))
# banana and monkey: 0.4041501317354622

print(word3.similarity(word1))
# banana and cat: 0.22358827466989753
print()


# It is interesting to see the most similar words are cat and monkey,
# which is probably because they are both animals, and also that banana and
# monkey is more similar than cat and banana, because monkeys eat bananas

# another example:
word1 = nlp("ball")
word2 = nlp("foot")
word3 = nlp("ear")
print(word1.similarity(word2))
# ball and foot: 0.38922090154899097
print(word3.similarity(word2))
# ear and foot: 0.25705044914570047
print(word3.similarity(word1))
# ear and ball: 0.2429078403416783

# ball and foot are the most similar, which means the model
# knows about the relationship between them (i.e. kicking a ball with your foot)


#########################################################

print()
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
    print()


# Again it is interesting to see cat and monkey being noted as being similar,
# and monkey and banana is thought to be more similar than monkey and apple.
# Also interesting that it does not seem to recognise transitive relationships,
# as in cat being similar to monkey does not make it similar to banana



#########################################################

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
print(f"model sentence: {model_sentence}\n")
for sentence in sentences:
      similarity = nlp(sentence).similarity(model_sentence)
      print(sentence + " - ",  similarity)


# it is intesting that the most similar sentence to the model is
# "Hello, there is my car", which means the model knows they both refer 
# to the car of the person speaking.
# Also the sentences refereing to cars and cats also got high similarity
# as they both refer to pets



####################################

# Running the example.py with the simpler language model does not give the
# same quality of similarity results as it does not come with word vectors so it
# can only use context-senitive tensors