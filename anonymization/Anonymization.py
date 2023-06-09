from collections import defaultdict
from typing import Iterable, Pattern, Callable, List, Any
import re

from faker import Factory

from .lib.diff_match_patch import diff_match_patch

class Anonymization:
    '''
    Faker wrapper providing utility functions, to map values with fakes equivalents.
    '''

    def __init__(self, locale: str):
        self.locale = locale
        self.faker = Factory.create(locale)
        self.anonDicts = {}

    def getFake(self, provider: str, match: str) -> str:
        '''
        Return the fake equivalent of match using a Faker provider
        
        Example:
                getFake(provider="date", match="09/02/1990") -> "2023-01-06"        
        '''
        if not provider in self.anonDicts:
            self.anonDicts[provider] = defaultdict(getattr(self.faker, provider))
        
        return self.anonDicts[provider][match] #Element is visible only when specific key value is returned (Refer to https://www.digitalocean.com/community/tutorials/python-getattr)
    
    def getPredefine(self, provider: str, match: str) -> str:
        '''
        Return the fake equivalent of match using a Faker provider
        
        Example:
                getFake(provider="date", match="09/02/1990") -> "2023-01-06"        
        '''
        if not provider in self.anonDicts:
            self.anonDicts[provider] = defaultdict(getattr(self.faker, provider))
        
        # print(f'{provider} - {match}')
        if provider == 'random_number' or provider == 'phone_number' or  provider == 'credit_card_number' or  provider == 'iban':
            self.anonDicts['number'] = self.anonDicts[provider]
            self.anonDicts['number'][match] = "[number_removed]"
        else:
            self.anonDicts[provider][match] = f"[{provider}_removed]"
        return self.anonDicts[provider][match] #Element is visible only when specific key value is returned (Refer to https://www.digitalocean.com/community/tutorials/python-getattr)

    def replace_all(self, text: str, matchs: Iterable[str], provider: str) -> str:
        '''
        Replace all occurance in matchs in text using a Faker provider
        
        Example:
                replace_all(text="My DOB is 09/02/1990, matchs=['09/02/1990']", provider="date") -> "My DOB is 2023-01-06"
        '''
        # Use fake entities for replacement:
        #for match in matchs:
        #    text = text.replace(match, f"{self.getFake(provider, match)}")
        # Use fixed regular expression for replacement:
        for match in matchs:
            # text = text.replace(match, self.getPredefine(provider, match))
            text = text.replace(match, f"<{provider.upper()}>")

        return text
    
    def regex_anonymizer(self, text: str, regex: Pattern, provider: str) -> str:
        '''
        Anonymize all substring matching a specific regex using a Faker provider
        '''
        matchs = re.findall(regex, text)
        return self.replace_all(text, matchs, provider)

    def add_provider(self, provider):
        '''
        Add a faker provider
        '''
        return self.faker.add_provider(provider)


class AnonymizerChain:
    '''
    Tool to run many anonymizers using a single anonymization context
    '''

    def __init__(self, anonymization: Anonymization):
        self.anonymization = anonymization
        self._anonymizers = []

    def add_anonymizers(self, *args: Iterable[Callable[[Anonymization], Any]]) -> None:
        '''
        Add one or many anonymizers
        '''
        for arg in args:
            self._anonymizers.append(arg(self.anonymization))

    def clear_anonymizers(self) -> None:
        '''
        Remove all anonymizers
        '''
        self._anonymizers = []

    def anonymize(self, text: str) -> str:
        '''
        Run all registered anonymizers on a text
        '''
        for anonymizer in self._anonymizers:
            text = anonymizer.anonymize(text)

        return text

    def anonymize_all(self, texts: Iterable[str]) -> List[str]:
        '''
        Run all registered anonymizers on a list of texts
        '''
        return [self.anonymize(text) for text in texts]
    
    def evaluate(self, text: str) -> str:
        '''
        Evaluate all registered anonymizers on a text
        '''
        result = []

        for anonymizer in self._anonymizers:
            result += anonymizer.evaluate(text)

        return result

    def pseudonymize(self, text: str) -> str:
        '''
        Run all registered anonymizes on a text and return also the diff patch
        '''
        dmp = diff_match_patch()
        clean = self.anonymize(text)
        diff = dmp.diff_main(clean, text)
        patch = dmp.patch_make(clean, text, diff)

        return clean, patch
    
    def revert(self, text: str, patch: str) -> str:
        '''
        Apply a patch on a cleaned text to revert the changes
        '''
        dmp = diff_match_patch()
        text, _ = dmp.patch_apply(patch, text)

        return text
