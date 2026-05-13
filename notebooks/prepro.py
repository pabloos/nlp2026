# import spacy

# nlp = spacy.load("en_core_web_sm")

# text = "I love natural language processing"

# doc = nlp(text)

# tokens = [token.lemma_ for token in doc]

# print(tokens)

# import spacy

# nlp = spacy.load("en_core_web_sm")

# text = "The easiest way to learn NLP is by doing projects and practicing regularly."

# doc = nlp(text)

# for token in doc:
#     print(
#         token.text,
#         token.lemma_,
#         token.pos_,
#         token.dep_
#     )


# eliminación de stopwords
# import spacy

# nlp = spacy.load("en_core_web_sm")

# text = "This is a very good phone"

# doc = nlp(text)

# tokens = [
#     token.text
#     for token in doc
#     if not token.is_stop
# ]

# print(tokens)

# import spacy
# from spacy import displacy

# nlp = spacy.load("en_core_web_sm")

# text = "Apple released the new iPhone in California."

# doc = nlp(text)

# displacy.serve(doc, style="dep", port=5001)

import spacy

texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    print([(ent.text, ent.label_) for ent in doc.ents])