from ir_system import *


def test_case():
    irs = InformationRetrievalSystem()
    while True: 
    # Step 1: Set the Model
        print("Step 1: Set the Retrieval Model")
        print('Available models:')
        print(f'{MODEL_BOOL_LIN} - Boolean model with linear search')
        print(f'{MODEL_BOOL_INV} - Boolean model with inverted lists')
        print(f'{MODEL_BOOL_SIG} - Boolean model with signature-based search')
        print(f'{MODEL_VECTOR} - Vector space model')
        model_choice = int(input('Enter the model choice: '))
        
        if model_choice == MODEL_BOOL_LIN:
            irs.model = models.LinearBooleanModel()
        elif model_choice == MODEL_BOOL_INV:
            irs.model = models.InvertedListBooleanModel()
        elif model_choice == MODEL_BOOL_SIG:
            irs.model = models.SignatureBasedBooleanModel()
        elif model_choice == MODEL_VECTOR:
            irs.model = models.VectorSpaceModel()
        else:
            print('Invalid choice. Exiting test case.')
            return
    
        print(f'You have selected: {irs.model}\n')
    
        # Step 2: Search for a term
        print("Step 2: Search for a Term")
        
        # Determine desired search parameters
        SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
        print('Search options:')
        print(f'{SEARCH_NORMAL} - Standard search (default)')
        print(f'{SEARCH_SW} - Search documents with removed stopwords')
        print(f'{SEARCH_STEM} - Search documents with stemmed terms')
        print(f'{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms')
        
        search_mode = int(input('Enter search mode choice: '))
        stop_word_filtering = (search_mode == SEARCH_SW) or (search_mode == SEARCH_SW_STEM)
        stemming = (search_mode == SEARCH_STEM) or (search_mode == SEARCH_SW_STEM)
        
        # Enter the query
        query = input('Enter the search query: ').lower()
        
        if stemming:
            query = porter.stem_query_terms(query)
    
        # Start query processing and measure time
        start_time = time.time()
    
        if isinstance(irs.model, models.InvertedListBooleanModel):
            results = irs.inverted_list_search(query, stemming, stop_word_filtering)
        elif isinstance(irs.model, models.VectorSpaceModel):
            results = irs.buckley_lewit_search(query, stemming, stop_word_filtering)
        elif isinstance(irs.model, models.SignatureBasedBooleanModel):
            results = irs.signature_search(query, stemming, stop_word_filtering)
        else:
            results = irs.basic_query_search(query, stemming, stop_word_filtering)
    
        end_time = time.time()
    
        # Output the results
        print(f'\nTotal results: {len(results)}\n')  # Show total number of results
        for index, (score, document) in enumerate(results, start=1):  # Enumerate to number the results
            print(f' {score}: {document}')
    
        # Output precision, recall, and query processing time
        print()
        print(f'Precision: {irs.calculate_precision(query, results)}')
        print(f'Recall: {irs.calculate_recall(query, results)}')
        print(f'Query processing time: {(end_time - start_time) * 1000} ms')  # Print time in ms
    

if __name__ == '__main__':
    test_case()
