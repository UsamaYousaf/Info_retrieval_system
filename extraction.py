# Contains functions that deal with the extraction of documents from a text file (see PR01)

import json

from document import Document


def extract_collection(source_file_path: str) -> list[Document]:
    """
    Loads a text file (aesopa10.txt) and extracts each of the listed fables/stories from the file.
    :param source_file_path: File name of the file that contains the fables
    :return: List of Document objects
    """
    with open(source_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define the start marker for the fables section
    fable_start_marker = "Aesop's Fables\n\n\n\n  The Cock and the Pearl"

    # Find the start index of the first fable
    fable_start_index = content.find(fable_start_marker)
    if fable_start_index == -1:
        raise ValueError("Start marker for fables not found.")

    # Extract content starting from the fables section
    content = content[fable_start_index:]

    # Splitting by 4 newlines to separate each fable
    fables = content.split('\n\n\n\n')

    catalog = []
    for doc_id, fable in enumerate(fables):
        parts = fable.strip().split('\n\n')
        if len(parts) < 2:
            continue

        title = parts[0].strip()
        raw_text = ' '.join(parts[1:]).replace('\n', ' ').strip()

        document = Document()
        document.document_id = doc_id
        document.title = title
        document.raw_text = raw_text
        document.terms = raw_text.split()

        catalog.append(document)

    return catalog



def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (= a list of Document objects)
    :param file_path: Path of the JSON file
    """

    serializable_collection = []
    for document in collection:
        serializable_collection += [{
            'document_id': document.document_id,
            'title': document.title,
            'raw_text': document.raw_text,
            'terms': document.terms,
            'filtered_terms': document.filtered_terms,
            'stemmed_terms': document.stemmed_terms
        }]

    with open(file_path, "w") as json_file:
        json.dump(serializable_collection, json_file)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            document.title = doc_dict.get('title')
            document.raw_text = doc_dict.get('raw_text')
            document.terms = doc_dict.get('terms')
            document.filtered_terms = doc_dict.get('filtered_terms')
            document.stemmed_terms = doc_dict.get('stemmed_terms')
            collection += [document]

        return collection
    except FileNotFoundError:
        print('No collection was found. Creating empty one.')
        return []
