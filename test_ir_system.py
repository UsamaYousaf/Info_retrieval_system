import unittest
from unittest.mock import patch
import models
import extraction
import porter
import cleanup
from document import Document
from ir_system import InformationRetrievalSystem

class TestInformationRetrievalSystem(unittest.TestCase):

    @patch('builtins.input', side_effect=['3', 'y', 'y', '5', '2', 'quick brown fox'])
    @patch('builtins.print')
    def test_pr03_task_1(self, mock_print, mock_input):
        # Initialize the system
        irs = InformationRetrievalSystem()
        
        # Set up the collection path and raw data path
        raw_collection_file = os.path.join(irs.RAW_DATA_PATH, 'aesopa10.txt')
        
        # Mock the extraction process
        irs.collection = extraction.extract_collection(raw_collection_file)
        
        # Ensure that documents are loaded
        self.assertTrue(isinstance(irs.collection, list))
        self.assertTrue(all(isinstance(d, Document) for d in irs.collection))
        
        # Filter stopwords
        cleanup.filter_collection(irs.collection, irs.STOPWORD_FILE_PATH)
        
        # Perform stemming
        porter.stem_all_documents(irs.collection)
        
        # Save the collection
        extraction.save_collection_as_json(irs.collection, irs.COLLECTION_PATH)
        
        # Set the retrieval model to InvertedListBooleanModel
        irs.model = models.InvertedListBooleanModel()
        
        # Build the inverted list
        irs.model.build_inverted_list(irs.collection)
        
        # Perform a search
        query = 'quick brown fox'
        results = irs.inverted_list_search(query, stemming=True, stop_word_filtering=True)
        
        # Check that results are returned
        self.assertTrue(len(results) > 0)
        
        # Print the results
        for score, document in results:
            mock_print.assert_any_call('{}: {}'.format(score, document))


    @patch('builtins.input', side_effect=['2', 'fox'])
    @patch('builtins.print')
    def test_pr03_task_2(self, mock_print, mock_input):
        # Initialize the system
        irs = InformationRetrievalSystem()
        
        # Load the existing collection
        irs.collection = extraction.load_collection_from_json(irs.COLLECTION_PATH)
        
        # Ensure that documents are loaded
        self.assertTrue(isinstance(irs.collection, list))
        self.assertTrue(all(isinstance(d, Document) for d in irs.collection))
        
        # Set the retrieval model to InvertedListBooleanModel
        irs.model = models.InvertedListBooleanModel()
        
        # Build the inverted list
        irs.model.build_inverted_list(irs.collection)
        
        # Perform a search
        query = 'fox'
        results = irs.inverted_list_search(query, stemming=False, stop_word_filtering=False)
        
        # Check that results are returned
        self.assertTrue(len(results) > 0)
        
        # Print the results
        for score, document in results:
            mock_print.assert_any_call('{}: {}'.format(score, document))


if __name__ == '__main__':
    unittest.main()
