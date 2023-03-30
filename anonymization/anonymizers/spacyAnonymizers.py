from types import SimpleNamespace

import spacy

from ..Anonymization import Anonymization

class _NamedEntitiesAnonymizer():
    '''
    Replace all named entities with fake ones

    This class requires spacy and a spacy model:
    $ pip install spacy
    $ python -m spacy download <model>

    Call NamedEntitiesAnonymizer if you want to pass an instance to an AnonymizerChain
    '''

    def __init__(self, anonymization: Anonymization, model: str):
        self.anonymization = anonymization
        self.processor = spacy.load(model)
        
    def has_numbers(self, inputString):
        # Check if a string contains a number
        return any(char.isdigit() for char in inputString)

    def anonymize(self, text: str) -> str:
        doc = self.processor(text)
        
        # remove whitespace entities and trim the entities
        ents = [ent.text.strip() for ent in doc.ents if not ent.text.isspace()]
        labels = [ent.label_ for ent in doc.ents if not ent.text.isspace()]
        
        # Added by MX
        entity_dic = {
            'PERSON': 'name',
            'NORP': 'country',
            'FAC': 'building_number',
            #'ORG': 'company',
            'GPE': 'address',
            'LOC': 'address',
            'DATE': 'date',
            'TIME': 'date',
            'PERCENT': 'random_number',
            'MONEY': 'random_number',
            'QUANTITY': 'random_number',
            'CARDINAL': 'random_number',
        } 
        
        for idx, ent in enumerate(ents):
            if labels[idx] in entity_dic:
                if labels[idx] in ['DATE', 'TIME'] and not self.has_numbers(ent):
                    continue
                else:
                    text = self.anonymization.replace_all(text, [ent], entity_dic[labels[idx]])

        return text #self.anonymization.replace_all(text, ents, 'first_name')
    
    def evaluate(self, text: str) -> str:
        doc = self.processor(text)
        # remove whitespace entities and trim the entities
        ents = [SimpleNamespace(start=ent.start_char, end=ent.end_char, entity_type=ent.label_, score=1) for ent in doc.ents if not ent.text.isspace()]

        return ents

def NamedEntitiesAnonymizer(model: str) -> _NamedEntitiesAnonymizer:
    '''
    Context wrapper for _NamedEntitiesAnonymizer, takes a spacy model.
    '''

    return lambda anonymization: _NamedEntitiesAnonymizer(anonymization, model)