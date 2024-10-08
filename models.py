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

from collections import Counter
import re

class LinearBooleanModel(RetrievalModel):
    def __init__(self):
        self.documents = []  # Store all documents

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into a representation suitable for Boolean retrieval.
        """
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        term_counter = Counter(term.lower() for term in terms)  # Convert terms to lower case
        return {term: 1 for term in term_counter}

    def query_to_representation(self, query: str):
        """
        Converts a query into a list of tokens, splitting on logical operators.
        """
        tokens = re.split(r'(\(|\)|\&|\||\-)', query.lower())  # Split by AND, OR, NOT operators and parentheses
        return [token.strip() for token in tokens if token.strip()]  # Remove any empty tokens or spaces

    def match(self, document_representation, query_representation) -> float:
       # Ensure both representations are dictionaries or convert them if they are lists
       if isinstance(document_representation, list):
           document_representation = {term: 1 for term in document_representation}  # Convert to dictionary
       
       if isinstance(query_representation, list):
           query_representation = {term: 1 for term in query_representation}  # Convert to dictionary
   
       # Now proceed with matching
       common_terms = set(document_representation.keys()).intersection(query_representation.keys())
       return len(common_terms)  # Return the count of common terms
   

    def search(self, query: str):
        """
        Searches for documents that match the given query, handling logical operators.
        """
        tokens = self.query_to_representation(query)  # Tokenize the query
        result_stack = []
        operator_stack = []  # Stack to hold operators like &, |, and -

        def apply_operator():
            """ Helper function to apply the operator on top of the operator stack to the result stack. """
            if len(result_stack) < 2 and operator_stack[-1] != '-':
                return set()  # Not enough operands for binary operators
            right = result_stack.pop() if operator_stack[-1] != '-' else None
            left = result_stack.pop() if right else result_stack.pop()

            operator = operator_stack.pop()

            if operator == '&':
                result_stack.append(left.intersection(right))  # AND operation
            elif operator == '|':
                result_stack.append(left.union(right))  # OR operation
            elif operator == '-':
                result_stack.append(self._negate_set(left))  # NOT operation (negation of left)

        for token in tokens:
            if token == '&' or token == '|':
                # Apply the last operator if it's already present
                while operator_stack and operator_stack[-1] in ('&', '|'):
                    apply_operator()
                operator_stack.append(token)
            elif token == '-':
                # Negation (NOT) operator; no need to check stack size before
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)  # Push the opening parenthesis
            elif token == ')':
                # Apply all operators until the matching '(' is found
                while operator_stack and operator_stack[-1] != '(':
                    apply_operator()
                operator_stack.pop()  # Pop the '(' from the stack
            else:
                # It's a term, get the set of document IDs containing the term
                result_stack.append(self._get_matching_docs(token))

        # After all tokens, apply any remaining operators
        while operator_stack:
            apply_operator()

        # Final check after processing all tokens
        if not result_stack:
            return []  # If nothing is on the stack, return no matches

        final_result = result_stack.pop()
        return [self.documents[doc_id] for doc_id in final_result]  # Return the documents that matched

    def _get_matching_docs(self, term):
        """
        Returns the set of document IDs that contain the given term.
        """
        matching_docs = set()
        for doc_id, doc_rep in enumerate(self.documents):
            if term in doc_rep:
                matching_docs.add(doc_id)
        return matching_docs

    def _negate_set(self, matching_docs):
        """
        Returns the complement of the matching documents (i.e., all documents that don't match).
        """
        all_docs = set(range(len(self.documents)))  # All document IDs
        return all_docs - matching_docs  # Documents that don't match

    def add_document(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Adds a document to the collection.
        """
        doc_rep = self.document_to_representation(document, stopword_filtering, stemming)
        self.documents.append(doc_rep)

    def __str__(self):
        return 'Boolean Model (Linear)'

class InvertedListBooleanModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = {}
        self.all_docs = set()

    def build_inverted_list(self, documents):
        """
        Builds an inverted list (index) from the given collection of documents.
        """
        for doc_id, document in enumerate(documents):
            self.all_docs.add(doc_id)
            for term in document.terms:
                term = term.lower()  # Convert term to lower case
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into a representation (just adds terms to the inverted list).
        """
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        term_counter = Counter(term.lower() for term in terms)  # Convert terms to lower case
        return {term: 1 for term in term_counter}  # Each term is represented by 1 (binary presence)

    def query_to_representation(self, query: str):
        """
        Converts a query into a tokenized form, splitting on logical operators.
        """
        tokens = re.split(r'(\(|\)|\&|\||\-)', query.lower())  # Split by AND, OR, NOT operators and parentheses
        return [token.strip() for token in tokens if token.strip()]  # Remove any empty tokens or spaces

    def match(self, document_representation, query_representation) -> float:
        """
        The match function for Boolean retrieval isn't a similarity measure; 
        it checks if a document contains the required terms.
        """
        doc_terms = set(document_representation.keys())  # Terms in the document
        query_terms = set(query_representation.keys())  # Terms in the query
        
        # Check if all query terms are present in the document (AND operation)
        return 1.0 if query_terms.issubset(doc_terms) else 0.0

    def search(self, query):
        """
        Searches for documents that match the given query, handling logical operators.
        """
        tokens = self.query_to_representation(query.lower())  # Tokenize the query
        result_stack = []
        operator_stack = []  # Stack to hold operators like &, |, and -
    
        def apply_operator():
            """ Helper function to apply the operator on top of the operator stack to the result stack. """
            if len(result_stack) < 2 and operator_stack[-1] != '-':
                return []  # Not enough operands for binary operators
            right = result_stack.pop() if operator_stack[-1] != '-' else None
            left = result_stack.pop() if right else result_stack.pop()
            
            operator = operator_stack.pop()
    
            if operator == '&':
                result_stack.append(left & right)
            elif operator == '|':
                result_stack.append(left | right)
            elif operator == '-':
                result_stack.append(self.all_docs - left)
    
        for token in tokens:
            if token == '&' or token == '|':
                # Apply the last operator if it's already present
                while operator_stack and operator_stack[-1] in ('&', '|'):
                    apply_operator()
                operator_stack.append(token)
            elif token == '-':
                # Negation (NOT) operator; no need to check stack size before
                operator_stack.append(token)
            else:
                # It's a term, get the set of documents containing the term
                if token in self.inverted_index:
                    result_stack.append(self.inverted_index[token])
                else:
                    result_stack.append(set())  # Empty set if the term is not in the inverted index
    
        # After all tokens, apply any remaining operators
        while operator_stack:
            apply_operator()
    
        # Final check after processing all tokens
        if not result_stack:
            return []  # If nothing is on the stack, return no matches
    
        return result_stack.pop()
    
        def __str__(self):
            return 'Boolean Model (Inverted List)'
    
class SignatureBasedBooleanModel(RetrievalModel):
    def __init__(self, F=64, D=4):
        self.F = F  # Bit length of the signature
        self.D = D  # Number of hash functions to use
        self.documents = []
        self.signatures = []  # This will store a list of signatures for each document

    def _hash_function(self, term, seed):
        # Hashing function based on term and seed
        h = hashlib.sha256(term.encode('utf-8') + str(seed).encode('utf-8'))
        return int(h.hexdigest(), 16) % self.F

    def _create_signature(self, terms):
        # Create a signature based on a list of terms
        signature = [0] * self.F
        for term in terms:
            term = term.lower()  # Convert term to lower case
            for i in range(self.D):
                pos = self._hash_function(term, i)
                signature[pos] = 1
        return signature

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # Get the terms for the document based on filtering and stemming
        if stopword_filtering:
            terms = document.filtered_terms
        else:
            terms = document.terms

        if stemming:
            terms = document.stemmed_terms

        # Divide the document terms into sections of 5 terms each
        sections = [terms[i:i+5] for i in range(0, len(terms), 5)]
        doc_signatures = []
        for section in sections:
            signature = self._create_signature(section)
            doc_signatures.append(signature)

        # Store the document and its corresponding signatures
        self.documents.append(document)
        self.signatures.append(doc_signatures)

        return doc_signatures  # Return the list of signatures for the document

    def query_to_representation(self, query: str):
        query_terms = query.lower().split()  # Convert query terms to lower case
        # Divide the query terms into sections of 5 terms (or fewer)
        sections = [query_terms[i:i+5] for i in range(0, len(query_terms), 5)]
        query_signatures = [self._create_signature(section) for section in sections]
        return query_signatures  # Return a list of signatures for each section of the query

    def match(self, document_signatures, query_signatures) -> bool:
        # Stricter matching threshold to avoid too many false positives
        for doc_sig in document_signatures:
            for query_sig in query_signatures:
                matching_bits = sum(1 for doc_bit, query_bit in zip(doc_sig, query_sig) if doc_bit & query_bit)
                query_active_bits = sum(query_sig)  # Total number of '1's in the query signature
                if query_active_bits > 0 and matching_bits / query_active_bits >= 0.75:  # Tighten match threshold
                    return True
        return False

    def search(self, query: str):
        """
        Searches for documents that match the given query, handling logical operators.
        """
        # Tokenize the query by splitting on AND (&), OR (|), parentheses, and stripping tokens
        tokens = re.split(r'(\(|\)|\&|\|)', query.lower())
        tokens = [token.strip() for token in tokens if token.strip()]  # Remove empty spaces

        result_stack = []
        operator_stack = []  # Stack for operators (&, |, -)

        def apply_operator():
            """ Helper function to apply the operator on top of the operator stack to the result stack. """
            if len(result_stack) < 2 and operator_stack[-1] != '-':
                return []  # Not enough operands for binary operators
            right = result_stack.pop() if operator_stack[-1] != '-' else None
            left = result_stack.pop() if right else result_stack.pop()

            operator = operator_stack.pop()

            if operator == '&':
                result_stack.append([l & r for l, r in zip(left, right)])  # Bitwise AND operation
            elif operator == '|':
                result_stack.append([l | r for l, r in zip(left, right)])  # Bitwise OR operation
            elif operator == '-':
                result_stack.append([not l for l in left])  # Logical NOT

        for token in tokens:
            if token in ('&', '|'):
                # Apply the last operator if it's already present
                while operator_stack and operator_stack[-1] in ('&', '|'):
                    apply_operator()
                operator_stack.append(token)
            elif token == '-':
                # Negation (NOT) operator; no need to check stack size before
                operator_stack.append(token)
            else:
                # It's a term, get the document signatures and perform matching
                query_signatures = self.query_to_representation(token)
                matches = [self.match(doc_signatures, query_signatures) for doc_signatures in self.signatures]
                result_stack.append(matches)

        # After all tokens, apply any remaining operators
        while operator_stack:
            apply_operator()

        # Final check after processing all tokens
        if not result_stack:
            return []  # If nothing is on the stack, return no matches

        final_matches = result_stack.pop()

        # Get all documents that matched
        results = [self.documents[i] for i, match in enumerate(final_matches) if match]

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