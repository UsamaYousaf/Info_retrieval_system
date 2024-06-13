# Contains all retrieval models.

from abc import ABC, abstractmethod
from document import Document
from collections import Counter
import re

class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """


class LinearBooleanModel(RetrievalModel):
    def __init__(self):
        pass  # No specific initialization needed for this model

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        # Create a binary vector indicating the presence/absence of terms
        term_counter = Counter(terms)
        return {term: 1 for term in term_counter}

    def query_to_representation(self, query: str):
        # Simple representation: binary vector indicating the presence/absence of terms
        query_terms = query.lower().split()  # Split query into terms
        return {term: 1 for term in query_terms}

    def match(self, document_representation, query_representation) -> float:
        # Simple matching: count the number of common terms between document and query
        common_terms = set(document_representation.keys()).intersection(query_representation.keys())
        return len(common_terms)  # Return the count of common terms

    def __str__(self):
        return 'Boolean Model (Linear)'


class InvertedListBooleanModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = {}
        self.all_docs = set()

    def build_inverted_list(self, documents):
        for doc_id, document in enumerate(documents):
            self.all_docs.add(doc_id)
            for term in document.terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        # Create a binary vector indicating the presence/absence of terms
        term_counter = Counter(terms)
        return {term: 1 for term in term_counter}

    def query_to_representation(self, query: str):
        # Parse the query into tokens
        tokens = re.split(r'(\(|\)|\&|\||\-)', query)
        return [token for token in tokens if token.strip()]

    def match(self, document_representation, query_representation) -> float:
        raise NotImplementedError()  # Not used in this model

    def search(self, query):
        tokens = self.query_to_representation(query)
        result_stack = []
        for token in tokens:
            if token == '&':
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append(left & right)
            elif token == '|':
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append(left | right)
            elif token == '-':
                term = result_stack.pop()
                result_stack.append(self.all_docs - term)
            else:
                if token in self.inverted_index:
                    result_stack.append(self.inverted_index[token])
                else:
                    result_stack.append(set())
        return result_stack.pop()

    def __str__(self):
        return 'Boolean Model (Inverted List)'


class SignatureBasedBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Boolean Model (Signatures)'


class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Vector Space Model'


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'
