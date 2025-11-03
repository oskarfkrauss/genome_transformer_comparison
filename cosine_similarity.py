from sklearn.metrics.pairwise import cosine_similarity

# data from Jonah
staph_person_1 = 'ACTGTTTGGAACCC'
staph_person_2 = 'ACTGTTTGGATTTT'

# embedding_person_1 = transformer_2(staph_person_1)
# embedding_person_2 = transformer_2(staph_person_2)

# turn these samples into a list of numbers
embedding_person_1 = [[1, 2, 3], [4, 5, 6]]
embedding_person_2 = [[6, 2, 3], [4, 5, 1]]

# how similar are these two lists of numbers
metric = cosine_similarity(embedding_person_1, embedding_person_2)
print("hello")
