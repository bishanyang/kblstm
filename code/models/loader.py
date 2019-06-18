
def create_mapping_with_unk(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    
    id_to_word = {index + 1: w[0] for (index, w) in enumerate(sorted_items)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    
    id_to_word[0] = "<unk>"
    word_to_id["<unk>"] = 0
    return word_to_id, id_to_word

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def lookup_word(word, word_to_lemmas, pretrained):
    if word in pretrained:
        return word
    elif word.lower() in pretrained:
        return word.lower()
    elif word in word_to_lemmas:
        for word in word_to_lemmas[word]:
            if word in pretrained:
                return word
            elif word.lower() in pretrained:
                return word.lower()
    return ""

def augment_with_pretrained(dictionary, word_to_id, id_to_word, pretrained, word_to_lemmas):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    for word in word_to_lemmas:
        if word not in dictionary:
            hit_word = lookup_word(word, word_to_lemmas, pretrained)
            if hit_word != "": 
                dictionary[word] = 0
                wid = len(word_to_id)
                word_to_id[word] = wid
                id_to_word[wid] = word
    
