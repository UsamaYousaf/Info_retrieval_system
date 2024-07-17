import re
from document import Document

# Helper functions to identify certain conditions in stems
def get_measure(term: str) -> int:
    """
    Returns the measure m of a given term [C](VC){m}[V].
    :param term: Given term/word
    :return: Measure value m
    """
    constants = "[^aeiou]"  
    vowels = "[aeiouy]"  
    form = f"^{constants}*({vowels}{constants})"
    m = len(re.findall(form, term))
    return m

def condition_v(stem: str) -> bool:
    """
    Returns whether condition *v* is true for a given stem (= the stem contains a vowel).
    :param stem: Word stem to check
    :return: True if the condition *v* holds
    """
    return bool(re.search(r"[aeiouy]", stem))

def condition_d(stem: str) -> bool:
    """
    Returns whether condition *d is true for a given stem (= the stem ends with a double consonant (e.g. -TT, -SS)).
    :param stem: Word stem to check
    :return: True if the condition *d holds
    """
    return bool(re.search(r"(.)\1$", stem))

def cond_o(stem: str) -> bool:
    """
    Returns whether condition *o is true for a given stem (= the stem ends cvc, where the second c is not W, X or Y
    (e.g. -WIL, -HOP)).
    :param stem: Word stem to check
    :return: True if the condition *o holds
    """
    return bool(re.search(r"[^aeiou][aeiouy][^aeiou][^wxy]$", stem))

def stem_term(term: str) -> str:
    """
    Stems a given term of the English language using the Porter stemming algorithm.
    :param term:
    :return:
    """
    original_term = term
    # Step 1a
    if term.endswith('sses'):
        term = term[:-2]
    elif term.endswith('ies'):
        term = term[:-2]
    elif term.endswith('ss'):
        pass
    elif term.endswith('s'):
        term = term[:-1]

    # Step 1b
    if term.endswith('eed'):
        if get_measure(term[:-3]) > 0:
            term = term[:-1]
    elif term.endswith('ed'):
        if condition_v(term[:-2]):
            term = term[:-2]
            if term.endswith(('at', 'bl', 'iz')):
                term += 'e'
            elif condition_d(term) and not term[-1] in "lsz":
                term = term[:-1]
            elif cond_o(term):
                term += 'e'
    elif term.endswith('ing'):
        if condition_v(term[:-3]):
            term = term[:-3]
            if term.endswith(('at', 'bl', 'iz')):
                term += 'e'
            elif condition_d(term) and not term[-1] in "lsz":
                term = term[:-1]
            elif cond_o(term):
                term += 'e'

    # Step 1c
    if term.endswith('y') and condition_v(term[:-1]):
        term = term[:-1] + 'i'

    # Step 2
    step_2_suffixes = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
        ('izer', 'ize'), ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'),
        ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
        ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
        ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble')
    ]
    for suffix, replacement in step_2_suffixes:
        if term.endswith(suffix) and get_measure(term[:-len(suffix)]) > 0:
            term = term[:-len(suffix)] + replacement
            break

    # Step 3
    step_3_suffixes = [
        ('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'),
        ('ical', 'ic'), ('ful', ''), ('ness', '')
    ]
    for suffix, replacement in step_3_suffixes:
        if term.endswith(suffix) and get_measure(term[:-len(suffix)]) > 0:
            term = term[:-len(suffix)] + replacement
            break

    # Step 4
    step_4_suffixes = [
        'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent',
        'sion', 'tion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
    ]
    for suffix in step_4_suffixes:
        if term.endswith(suffix) and get_measure(term[:-len(suffix)]) > 1:
            term = term[:-len(suffix)]
            break

    # Step 5a
    if term.endswith('e'):
        if get_measure(term[:-1]) > 1 or (get_measure(term[:-1]) == 1 and not cond_o(term[:-1])):
            term = term[:-1]

    # Step 5b
    if get_measure(term) > 1 and condition_d(term) and term.endswith('l'):
        term = term[:-1]

    # We read the porter.txt file and found the hidden note in line 354.

    return term

def stem_all_docs(doc_collection: list[Document]):
    """
    For each document in the given collection, this method uses the stem_term() function on all terms in its term list.
    Warning: The result is NOT saved in the document's term list, but in the extra field stemmed_terms!
    :param doc_collection: Document collection to process
    """
    for doc in doc_collection:
        doc.stemmed_terms = [stem_term(term) for term in doc.terms]

def stem_query_terms(query: str) -> str:
    """
    Stems all terms in the provided query string.
    :param query: User query, may contain Boolean operators and spaces.
    :return: Query with stemmed terms
    """
    terms = query.split()
    stemmed_terms = [stem_term(term) for term in terms]
    return ' '.join(stemmed_terms)
