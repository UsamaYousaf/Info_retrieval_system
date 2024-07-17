from abc import ABC, abstractmethod
from document import Document
from collections import Counter, defaultdict
import re
import hashlib
import numpy as np
import math

class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        pass

    @abstractmethod
    def query_to_representation(self, query: str):
        pass

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        pass

class LinearBooleanModel(RetrievalModel):
    def __init__(self):
        pass

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        term_counter = Counter(term.lower() for term in terms)  # Convert terms to lower case
        return {term: 1 for term in term_counter}

    def query_to_representation(self, query: str):
        query_terms = query.lower().split()  # Split query into terms and convert to lower case
        return {term: 1 for term in query_terms}

    def match(self, document_representation, query_representation) -> float:
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
                term = term.lower()  # Convert term to lower case
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

        term_counter = Counter(term.lower() for term in terms)  # Convert terms to lower case
        return {term: 1 for term in term_counter}

    def query_to_representation(self, query: str):
        tokens = re.split(r'(\(|\)|\&|\||\-)', query.lower())  # Convert query to lower case
        return [token for token in tokens if token.strip()]

    def match(self, document_representation, query_representation) -> float:
        raise NotImplementedError()

    def search(self, query):
        tokens = self.query_to_representation(query.lower())  # Convert query to lower case
        result_stack = []
        for token in tokens:
            if token == '&':
                if len(result_stack) < 2:  
                    return []
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append(left & right)
            elif token == '|':
                if len(result_stack) < 2:  
                    return []
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append(left | right)
            elif token == '-':
                if not result_stack:  # Ensure there is an operand
                    return []
                term = result_stack.pop()
                result_stack.append(self.all_docs - term)
            else:
                if token in self.inverted_index:
                    result_stack.append(self.inverted_index[token])
                else:
                    result_stack.append(set())
        if not result_stack:
            return []
        return result_stack.pop()

    def __str__(self):
        return 'Boolean Model (Inverted List)'

class SignatureBasedBooleanModel(RetrievalModel):
    def __init__(self, F=64, D=4):
        self.F = F
        self.D = D
        self.documents = []
        self.signatures = []

    def _hash_function(self, term, seed):
        h = hashlib.sha256(term.encode('utf-8') + str(seed).encode('utf-8'))
        return int(h.hexdigest(), 16) % self.F

    def _create_signature(self, terms):
        signature = [0] * self.F
        for term in terms:
            term = term.lower()  # Convert term to lower case
            for i in range(self.D):
                pos = self._hash_function(term, i)
                signature[pos] = 1
        return signature

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        signature = self._create_signature(terms)
        self.documents.append(document)
        self.signatures.append(signature)
        return signature

    def query_to_representation(self, query: str):
        query_terms = query.lower().split()  # Convert query terms to lower case
        query_signature = self._create_signature(query_terms)
        return query_signature

    def match(self, document_representation, query_representation) -> float:
        match_count = sum(1 for doc_bit, query_bit in zip(document_representation, query_representation) if doc_bit & query_bit)
        return match_count / self.F  # Normalize by the bit array length

    def search(self, query: str):
        tokens = re.split(r'(\(|\)|\&|\|)', query.lower())  # Convert query to lower case
        tokens = [token.strip() for token in tokens if token.strip()]

        result_stack = []
        for token in tokens:
            if token == '&':
                if len(result_stack) < 2:  
                    return []
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append([l & r for l, r in zip(left, right)])
            elif token == '|':
                if len(result_stack) < 2:  
                    return []
                right = result_stack.pop()
                left = result_stack.pop()
                result_stack.append([l | r for l, r in zip(left, right)])
            else:
                query_signature = self.query_to_representation(token)
                match_scores = [self.match(doc_sig, query_signature) for doc_sig in self.signatures]
                result_stack.append([score > 0 for score in match_scores])

        if not result_stack:
            return []
        results = [self.documents[i] for i, match in enumerate(result_stack.pop()) if match]
        return results

    def __str__(self):
        return 'Boolean Model (Signatures)'

class VectorSpaceModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.document_vectors = {}
        self.doc_lengths = {}
        self.num_documents = 0

    def build_inverted_index(self, documents):
        self.num_documents = len(documents)
        for doc_id, document in enumerate(documents):
            term_freq = Counter(term.lower() for term in document.terms)  # Convert terms to lower case
            for term, freq in term_freq.items():
                self.inverted_index[term].append((doc_id, freq))
        
        for doc_id, document in enumerate(documents):
            self.document_vectors[doc_id] = self._create_document_vector(doc_id, document.terms)
        
        self.doc_lengths = {doc_id: np.linalg.norm(vec) for doc_id, vec in self.document_vectors.items()}

    def _create_document_vector(self, doc_id, terms):
        term_freq = Counter(term.lower() for term in terms)  # Convert terms to lower case
        vec = np.zeros(len(self.inverted_index))
        for term, idx in zip(self.inverted_index.keys(), range(len(self.inverted_index))):
            if term in term_freq:
                tf = term_freq[term]
                idf = math.log(self.num_documents / len(self.inverted_index[term]))
                vec[idx] = tf * idf
        return vec

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        return self._create_document_vector(document.document_id, terms)

    def query_to_representation(self, query: str):
        query_terms = query.lower().split()  # Convert query terms to lower case
        term_freq = Counter(query_terms)
        vec = np.zeros(len(self.inverted_index))
        for term, idx in zip(self.inverted_index.keys(), range(len(self.inverted_index))):
            if term in term_freq:
                tf = term_freq[term]
                idf = math.log(self.num_documents / len(self.inverted_index[term]))
                vec[idx] = tf * idf
        return vec

    def match(self, document_representation, query_representation) -> float:
        if np.linalg.norm(document_representation) == 0 or np.linalg.norm(query_representation) == 0:
            return 0.0
        return np.dot(document_representation, query_representation) / (np.linalg.norm(document_representation) * np.linalg.norm(query_representation))

    def __str__(self):
        return 'Vector Space Model'

class FuzzySetModel(RetrievalModel):
    def __init__(self):
        raise NotImplementedError()

    def __str__(self):
        return 'Fuzzy Set Model'