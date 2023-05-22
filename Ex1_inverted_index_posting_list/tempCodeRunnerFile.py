inverted_index = create_inverted_index(docs)
sorted_vocabulary = get_sorted_vocabulary(docs)
query_with_inv_index(queries, inverted_index)