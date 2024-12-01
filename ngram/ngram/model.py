import pandas as pd
from tqdm import tqdm
import re
import random
from typing import List, defaultdict
from copy import copy
import random

def create(path, n=3):
    emails = pd.read_csv('emails.csv')
    print('File read')

    full_corpus = []
    for i in tqdm(range(len(emails))):
        email_text = emails['message'][i]
        cleaned_email = clean_email(email_text)
        if cleaned_email is not None:
            cleaned_email = re.sub(r"[^a-zA-Z.!? ]+", "", cleaned_email.replace("!", ".").replace("?", "."))
            tokens = re.findall(r'\w+|[^\w\s]', cleaned_email.lower(), re.UNICODE)
            if '@' in tokens:
                print('@')
            full_corpus.append(['@','@']+tokens)
    print('Corpus recorded')

    word_completor = WordCompletor(full_corpus)
    print('Completor is done')

    n_gram_model = NGramLanguageModel(corpus=full_corpus, n=n)
    print('Model is done')

    text_suggestion = TextSuggestion(word_completor, n_gram_model)
    print('GOOD')

    return text_suggestion

def inp_string(inp, n):
    inp_list = inp.split()
    if len(inp_list) < n:
        inp_list = ['@']*(n-len(inp_list)) + inp_list
    return inp_list

def out_string(inp, out):
    if len(out[0])==0:
        return ['']
    beg_of_last = inp[-1]
    bold = ' '.join(inp)
    prod = []
    for sp in out:
        full = sp[0]
        p = ' '.join(sp)
        if full == beg_of_last:
            text = re.sub(r"\s+\.", ".", ' '+p[len(full)+1:])
            prod.append(text)
        else:
            l = len(beg_of_last)
            text = re.sub(r"\s+\.", ".", p[l:])
            prod.append(text)
    return prod

def clean_email(content):
    # Если строка начинается с "To:", удаляем её и две предыдущие строки
    lines = content.split('\n')
    stop_list = []
    for i in range(len(lines)):  
        # Если строка начинается с "To:", удаляем её и две предыдущие строки
        if lines[-i-1].strip().startswith("To:"):
            stop_list.append(len(lines)-i-1)
            stop_list.append(len(lines)-i-2)
            stop_list.append(len(lines)-i-3)
    stop_list= sorted(list(set(stop_list)))
    clean_lines = [elem for i, elem in enumerate(lines) if i not in stop_list]

    # Собираем текст обратно в одну строку
    content = "\n".join(clean_lines).strip()
    
    # Удаляем строки с метаданными (Message-ID, Date, From, To, Subject и другие)
    content = re.sub(r"(?i)(Message-ID|Date|From|Subject|Mime-Version|Content-Type|cc|Content-Transfer-Encoding|X-From|X-To|X-cc|X-bcc|X-Folder|X-Origin|X-FileName):.*", "", content)
    content = re.sub(r"--", "", content)
    content = re.sub(r"==", "", content)
    content = re.sub(r"__", "", content)
    content = re.sub(r"\*{2}", "", content)
    content = re.sub(r"IMAGE", "", content)
    
    # Убираем строку с «Forwarded by» и подобные повторяющиеся сообщения
    content = re.sub(r"(?i)^(.*?Forwarded by.*?)$", "", content, flags=re.MULTILINE)

    # Удаляем строки с символом '@'
    content = re.sub(r"^.*@.*$", "", content, flags=re.MULTILINE)

    # Убираем ненужные пустые строки (состоящие только из пробелов)
    content = re.sub(r"^\s*$", "", content, flags=re.MULTILINE)

    # Убираем все переносы строк, заменяя их на пробелы
    content = re.sub(r"\n", " ", content)
    content = re.sub(r'\s+', ' ', content)

    if len(content) == 0:
        return None

    if content[0] == ' ':
        content = content[1:]

    return content

class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False

class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        
        for word in tqdm(vocabulary):
            current_node = self.root
            for letter in word:
                if letter not in current_node.children:
                    current_node.children[letter] = PrefixTreeNode()
                current_node = current_node.children[letter]
            current_node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        current_node = self.root
        for letter in prefix:
            if letter not in current_node.children:
                return []  # Если префикс не найден
            current_node = current_node.children[letter]
            
        result = []
        self._find_words_with_prefix(current_node, prefix, result)
        return result

    def _find_words_with_prefix(self, node: PrefixTreeNode, prefix: str, result: List[str]):
        """
        Рекурсивно находим все слова от текущей вершины
        """
        if node.is_end_of_word:
            result.append(prefix)
        
        # Проходим по всем дочерним вершинам
        for letter, child_node in node.children.items():
            self._find_words_with_prefix(child_node, prefix + letter, result)

class WordCompletor:
    def __init__(self, corpus):
        """
        corpus: list – корпус текстов
        """
        self.volume, self.word_counts = self._get_word_counts(corpus) # Определяем общий объём слов и количество каждого уникального слова
        self.prefix_tree = PrefixTree(list(self.word_counts.keys()))

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words = self.prefix_tree.search_prefix(prefix) # Слова, начинающиеся на префикс
        probs = [self.word_counts[i]/self.volume for i in words] # Вероятность = количество слова / общее количесвто слов
        
        return words, probs

    def _get_word_counts(self, corpus: List[List[str]]) -> dict:
        word_counts = defaultdict(int)
        volume = 0
        # Подсчитываем частоту появления слов в списках
        for doc in tqdm(corpus):
            seen_words = set()  # Чтобы слово не учитывалось несколько раз в одном документе
            volume += len(doc)
            for word in doc:
                if word not in seen_words:
                    word_counts[word] += 1
                    seen_words.add(word)
        return volume, word_counts

class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.n = n
        self.corpus = corpus
        
        # Словарь для хранения n-грамм: {префикс -> {следующее слово: частота}}
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.prefix_counts = defaultdict(int)
        
        # Строим модель на основе корпуса
        self._build_model()

    def _build_model(self):
        """
        Строит n-граммную модель на основе корпуса.
        """
        for sentence in tqdm(self.corpus):
            # Обрабатываем каждое предложение
            for i in range(len(sentence) - self.n):
                # Берем n-грамму из предложения
                prefix = tuple(sentence[i:i + self.n])  # (n-1)-грамма
                next_word = sentence[i + self.n]  # Следующее слово
                
                # Обновляем счетчик для префикса
                self.ngram_counts[prefix][next_word] += 1
                self.prefix_counts[prefix] += 1

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а также список вероятностей этих слов.
        """
        flag=0
        old_n = self.n
        if len(prefix) > self.n:
            prefix = prefix[-self.n:]
        elif len(prefix) < self.n:
            self.n = len(prefix)
            self._build_model()
            flag=1
            
        # Преобразуем prefix в кортеж
        prefix_tuple = tuple(prefix)
        
        # Получаем возможные следующие слова
        next_words = list(self.ngram_counts[prefix_tuple].keys())
        
        # Вычисляем вероятности
        probs = [
            count / self.prefix_counts[prefix_tuple]
            for count in self.ngram_counts[prefix_tuple].values()
        ]

        if flag:
            self.n = old_n
            self._build_model()
            
        return next_words, probs

class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model
        
    def inp2key(self, inp, l=10):
        k = {i:j for i,j in zip(*self.n_gram_model.get_next_words_and_probs(inp))}
        top_l = dict(sorted(k.items(), key=lambda item: item[1], reverse=True)[:l])
        return top_l
        
    def get_tree(self, inp, ns):
        f = self.inp2key(inp)
        s = {}
        for k in f.keys():
            s0 = self.inp2key(inp[1:]+[k])
            s0 = {tuple([k]+[key]): value*f[k] for key, value in s0.items()}
            s = s|s0
        for i in range(ns-2):
            f = s.copy()
            s = {}
            for k in f.keys():
                s0 = self.inp2key(tuple(inp[1:]+list(k)))
                s0 = {tuple(list(k)+[key]): value*f[k] for key, value in s0.items()}
                s = s|s0
        return s
        
    def suggest_text(self, text: list, n_words=3, n_texts=1) -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)
        
        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)
        
        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """
        suggestions = []
        
        words, probs = self.word_completor.get_words_and_probs(text[-1])
        if len(probs) == 0:
            return [[]]
        else:
            first = words[probs.index(max(probs))]

        out = []
        inp = text[:-1]+[first]
        for i in range(n_words):
            words, probs = self.n_gram_model.get_next_words_and_probs(inp)
            if len(probs) == 0:
                return [[]]
            else:
                pred = words[probs.index(max(probs))]
                out.append(pred)
                inp.append(pred)

        most = inp[-n_words-1:]
        suggestions.append(list(most))
        if n_texts > 1:
            sp = list(self.get_tree(inp, n_words).keys())
            sp = [[first]+list(t) for t in sp]
            if most in sp:
                sp.remove(most)
            for t in random.sample(sp, min(len(sp), n_texts-1)):
                suggestions.append(t)
        
        return suggestions