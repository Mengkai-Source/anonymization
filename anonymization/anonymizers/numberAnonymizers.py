import re
from types import SimpleNamespace

from ..Anonymization import Anonymization

class NumberAnonymizer():
    '''
    Replace the dates with fake ones

        Number Formats: lm1238****32 or 32****21 
        Reference: https://stackoverflow.com/questions/13453999/regex-replace-mixed-numberstrings
    '''

    def __init__(self, anonymization: Anonymization):
        self.anonymization = anonymization
        #self.number_regex = r"""(?x) # verbose regex
        #\b    # Start of word
        #(?=   # Look ahead to ensure that this word contains...
        #\w*  # (after any number of alphanumeric characters)
        #\d   # ...at least one digit.
        #)     # End of lookahead
        #\w+   # Match the alphanumeric word
        #\s*   # Match any following whitespace"""
        self.number_regex = r"""[a-zA-Z]+-+\d+|[a-zA-Z]+\d+""" # Capture patterns: Letters+-+Digits, Letter+Digits

    def anonymize(self, text: str) -> str:
        return  return  self.anonymization.regex_anonymizer(text, self.number_regex, 'port_number') # 'random_number'

    #def evaluate(self, text: str) -> str:
    #    matchs = re.finditer(self.number_regex, text)
    #   ents = [SimpleNamespace(start=m.start(), end=m.end(), entity_type="DATE", score=1) for m in matchs]
    #    return ents

