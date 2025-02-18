import re


def format_sentence(sentence):
    # Remove non-word characters except for spaces and hyphens
    clean_sentence = re.sub(r'[^\w\s-]', '', sentence)
    # Replace spaces with underscores
    underscore_sentence = re.sub(r'\s+', '_', clean_sentence)
    words = underscore_sentence.split('_')
    # Join the first 20 words with underscores
    limited_sentence = '_'.join(words[:20])
    return limited_sentence