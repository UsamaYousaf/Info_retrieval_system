# Contains all functions that deal with stop word removal.

from document import Document
from collections import Counter
import string

def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.

    :param text_string: The string to be cleaned.
    :return: The cleaned string.
    """
    # Remove "'s" occurrences
    cleaned_text = text_string.replace("'s", "")

    # Remove all punctuation marks
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

    return cleaned_text

def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    return term.lower() in stop_word_list

def remove_stop_words_from_term_list(term_list: list[str], stop_word_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :param stop_word_list: List of stop words
    :return: List of terms without stop words
    """
    cleaned_terms = []
    for term in term_list:
        term = remove_symbols(term)
        if not is_stop_word(term, stop_word_list):
            cleaned_terms.append(term)
    return cleaned_terms


def filter_collection(collection: list[Document], stop_word_list: list[str]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.

    :param collection: Document collection to process
    :param stop_word_list: List of stop words to filter out
    """
    for document in collection:
        document.filtered_terms = remove_stop_words_from_term_list(document.terms, stop_word_list)


def load_stop_word_list(raw_file_path: str) -> list[str]:
    """
    Loads a text file that contains stop words and saves it as a list. The text file is expected to be formatted so that
    each stop word is in a new line, e. g. like englishST.txt
    :param raw_file_path: Path to the text file that contains the stop words
    :return: List of stop words
    """
    # TODO: Implement this function. (PR02)
    with open(raw_file_path, 'r') as file:
        stop_word_list = [line.strip().lower() for line in file]
    return stop_word_list


def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    """
    Uses the method of J. C. Crouch (1990) to generate a stop word list by finding high and low frequency terms in the
    provided collection.
    :param collection: Collection to process
    :param high_frequency_threshold: Threshold for high frequency terms
    :param low_frequency_threshold: Threshold for low frequency terms
    :return: List of stop words
    """
    # Count term frequency in the collection
    term_counter = Counter()
    for doc in collection:
        term_counter.update(doc.terms)

    # Sort terms by frequency
    sorted_terms = sorted(term_counter.items(), key=lambda x: x[1], reverse=True)
    high_frequency_threshold = 100  # Example threshold for high frequency terms
    low_frequency_threshold = 5  # Example threshold for low frequency terms
    # Determine high and low frequency terms

    high_frequency_terms = [term for term, freq in sorted_terms if freq >= high_frequency_threshold]
    low_frequency_terms = [term for term, freq in sorted_terms if freq <= low_frequency_threshold]

    # Return low frequency terms as stop words
    return low_frequency_terms
