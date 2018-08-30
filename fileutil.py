
import textract
import re
import glob

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# global vars
default_files = ['doc', 'docx', 'pdf']


# this function parse resumes in various types such as .doc, .docx, .pdf.
def resume2text(file_path):
    content = textract.process(file_path)

    # decode content into Unicode
    content = str(content, 'utf-8')
    # split content into lines
    lines = content.split('\n')
    # Concatenate all lines into one paragraph
    paragraph = ''
    for line in lines:
        if line != '':
            paragraph += line + '\n'
    return paragraph


def extract_keywords(text, stemming=False):
    text = text.lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('[\d]', '', text)

    # The word_tokenize() function will break our text phrases into individual words
    tokens = word_tokenize(text)

    # we'll create a new list which contains punctuation we wish to clean
    # punctuations = ['(', ')', ';', ':', '[', ']', ',', '.', '!', '|', '@', '*', '-', '#']

    # We initialize the stopwords variable which is a list of words like
    # "The", "I", "and", etc. that don't hold much value as keywords
    stop_words = stopwords.words('english')

    # We create a list comprehension which only returns a list of words
    # that are NOT IN stop_words and NOT IN punctuations.
    keywords = [word for word in tokens if not word in stop_words]

    # Stemming the words
    if stemming:
        stemmer = PorterStemmer()
        keywords = list(map(lambda t: stemmer.stem(t), keywords))
    return keywords


# def read_resumes(path_to_folder, file_types = ['docx', 'pdf'], join = False):
#     texted_files = []
#     for file_type in file_types:
#         file_path = path_to_folder
#         file_path += '/*.' + file_type
#         files = glob.glob(file_path)
#         for file in files:
#             texted_files.append(resume2text(file))
#     if(join == True):
#        texted_files = ' '.join(texted_files)
#     return texted_files

def read_resumes(path_to_folder, file_types=default_files):
    texted_files = []
    file_names = []
    for file_type in file_types:
        file_path = path_to_folder
        file_path += '/*.' + file_type
        files = glob.glob(file_path)
        for file in files:
            try:
                text = resume2text(file)
            except:
                print('Fail to parse ' + file)
                continue
            text = text.strip()
            if len(text) == 0:
                print('Fail to parse ' + file)
            else:
                texted_files.append(text)
                file_names.append(file)
    return texted_files, file_names

def resumes2keywords(path_to_folder, file_types=default_files):
    texted_files, file_names = read_resumes(path_to_folder, file_types)
    keywords_list = []
    for texted in texted_files:
        keywords = extract_keywords(texted)
        keywords_list.append(keywords)
    return keywords_list, file_names


def resumes2data(label, path_to_folder, file_types=default_files):
    keywords, file_names = resumes2keywords(path_to_folder, file_types)
    labels = [label] * len(keywords)
    return keywords, labels, file_names


def resumes2textfile(label, path_to_folder, save_file_path, file_types=default_files):
    keywords, labels = resumes2data(label, path_to_folder, file_types)
    with open(save_file_path, 'w') as f:
        f.write('\n'.join('\t'.join([labels[i], ' '.join(keywords[i])]) for i in range(len(labels))))

