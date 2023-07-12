import re
import numpy as np
import string
import random
from datetime import datetime, timedelta
from faker import Factory
from presidio_analyzer import AnalyzerEngine
import us
import usaddress
from transformers import AutoTokenizer, TFBertForTokenClassification
import tensorflow as tf


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = TFBertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
fake = Factory.create()
analyzer = AnalyzerEngine()
us_states = [state.lower() for key, value in us.states.mapping('abbr', 'name').items() for state in [key, value]]


def getFake(provider: str) -> str:
    '''
    Return the fake equivalent of match using a Faker provider

    Example:
            getFake(provider="date", match="09/02/1990") -> "2023-01-06"        
    '''
    
    return getattr(fake, provider)()

def anonymize_credit_card(text: str) -> str:
    """
    Anonymize credit card
    
    Example:
            anonymize_credit_card('1232-3343-3443-4343') -> '9146-3972-9818-1319'
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
                
    return ''.join(list_text)

def anonymize_datetime(text: str, provider='date') -> str:
    """
    Anonymize datetime
    """
    
    # No Number
    if not re.search(r'\d+', text):
        return text
    
    # Handle false DATE_TIME label returned from presidio - No Number + Letters
    elif re.search(r'^\d+[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
        return ''.join(list_text)
    
    # No Letter and Punctuation
    elif not re.search(r'[a-zA-Z]+', text) and not re.search('\W+', text):
        return str(int(text) + np.random.randint(1, 3))
    
    else:
        return (datetime.now() - timedelta(days=np.random.randint(60, 700)) ).strftime('%Y-%m-%d')

def anonymize_email_address(text: str, provider='email') -> str:
    """
    Anonymize email address
    """
    if not re.search(r'[@]', text): # No letters or @ sign is not present
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters).upper()
        
        return ''.join(list_text)
    
    else:
        
        return getFake(provider)

def anonymize_iban_code(text: str) -> str:
    """
    Anonymize international bank account number
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_location(text: str) -> str:
    """
    Anonymize location: only change Street Number & Name
    """
    parse_address = [list(ele) for ele in usaddress.parse(text)]
    
    for index, parse_ele in enumerate(parse_address):
        if parse_ele[1] == 'AddressNumber':
            list_ele = list(parse_ele[0])
            for idx, ele in enumerate(list_ele):
                if ele.isdigit():
                    list_ele[idx] = str(np.random.randint(1, 10))
                elif ele.isalpha():
                    list_ele[idx] = random.choice(string.ascii_letters)
            parse_address[index][0] = ''.join(list_ele)
            
        elif parse_ele[1] == 'StreetName':
            parse_address[index][0] = getFake('street_name').split()[0]
            
        else:
            continue
    
    return ' '.join([i[0] for i in parse_address])

def augment_anonymize_location(text: str) -> str:
    """
    Detect location
    """
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

    # Tokens are classified rather then input words, so there are more predicted token classes than words. Multiple token classes might account for the same word
    predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    
    word_expand = list()
    for key, value in {x : tokenizer.encode(x, add_special_tokens=False) for x in text.split()}.items():
        word_expand.extend([key]*len(value))
    
    location_words = [word_expand[idx] for idx, ele in enumerate(predicted_tokens_classes) if 'LOC' in ele]
    
    for word in location_words:
        if word.lower() in us_states:
            continue
        else:
            text = text.replace(word, getFake('city').split()[0])
    
    return text

def anonymize_person(text: str, provider='name') -> str:
    """
    Anonymize name
    """
    # Handle false PERSON label returned from presidio
    if re.search(r'\d+', text) and not re.search(r'[a-zA-Z]+', text): # Just digits
        provider = 'date'
        
        return getFake(provider)
    
    elif re.search(r'\d+', text) and re.search(r'[a-zA-Z]+', text): # Digits + Letters
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            else:
                continue
        
        return ''.join(list_text)
    
    else:
        
        return getFake(provider)

def anonymize_phone_number(text: str) -> str:
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_medical_license(text: str) -> str:
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_url(text: str, provider='url') -> str:
    """
    Anonymize URL
    """
    
    return getFake(provider)

def anonymize_bank_number(text: str) -> str:
    """
    Anonymize US bank account
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_driver_license(text: str) -> str:
    """
    Anonymize driver license
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters).upper()
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_itin(text: str) -> str:
    """
    Anonymize taxplayer identification number (ITIN) similar to SSN
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters)
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(0, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_passport(text: str) -> str:
    """
    Anonymize passport number
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters).upper()
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_ssn(text: str) -> str:
    """
    Anonymize Social Security Number (SSN)
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters).upper()
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)

def anonymize_non_us(text: str) -> str:
    """
    Anonymize all Non-US entities returned from presidio (UK_NHS, ES_NIF, IT_FISCAL_CODE, etc.)
    Reference: https://microsoft.github.io/presidio/supported_entities/
    """
    if re.search(r'[a-zA-Z]+', text):
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            elif ele.isalpha():
                list_text[idx] = random.choice(string.ascii_letters).upper()
    else:
        list_text = list(text)
        for idx, ele in enumerate(list_text):
            if ele.isdigit():
                list_text[idx] = str(np.random.randint(1, 10))
            else:
                continue
    
    return ''.join(list_text)
    
    
def get_faker(entity_type):
    
    # https://microsoft.github.io/presidio/supported_entities/
    parse_dict = {
        # global
        'CREDIT_CARD': anonymize_credit_card,
        'DATE_TIME': anonymize_datetime,
        'EMAIL_ADDRESS': anonymize_email_address,
        'IBAN_CODE': anonymize_iban_code,
        #'IP_ADDRESS': anonymize_ip_address,
        #'NRP': anonymize_nrp,
        'LOCATION': anonymize_location,
        'PERSON': anonymize_person,
        'PHONE_NUMBER': anonymize_phone_number,
        'MEDICAL_LICENSE': anonymize_medical_license,
        'URL': anonymize_url,
        # USA
        'US_BANK_NUMBER': anonymize_bank_number,
        'US_DRIVER_LICENSE': anonymize_driver_license,
        'US_ITIN': anonymize_itin,
        'US_PASSPORT': anonymize_passport,
        'US_SSN': anonymize_ssn,
        'NON_US': anonymize_non_us,
    }

    return parse_dict.get(entity_type, NotImplemented)

def entity_anonymize(entity_text: str, analyzer_result: dict) -> str:
    entity_type = analyzer_result.get('entity_type')
    fake_maker = get_faker(entity_type)
    if fake_maker is NotImplemented:
        # raise NotImplementedError
        fake_maker = anonymize_non_us
        return fake_maker(entity_text)
    else:
        return fake_maker(entity_text)
    
def anonymize(text: str) -> str:
    """
    Anonymize text
    """
    anonDicts = {}
    analyzer_results = analyzer.analyze(text=text, language="en")
    analyzer_results = [analyzer_result.to_dict() for analyzer_result in analyzer_results]
    entity_text = [text[analyzer_result.get('start'):analyzer_result.get('end')] for analyzer_result in analyzer_results]
    
    for idx, analyzer_result in enumerate(analyzer_results):
        if entity_text[idx] not in anonDicts:
            anonDicts[entity_text[idx]] = analyzer_result.get('entity_type')
            text = text.replace(entity_text[idx], entity_anonymize(entity_text[idx], analyzer_result))
        else:
            continue
    
    # Further anonymize location/address
    text = augment_anonymize_location(text)
    
    return text