#!/usr/bin/env python3
import os
import time
import re


# -------------------------------------------------
# Part 1: Load Inverted Index and Build All Documents
# -------------------------------------------------
def load_inverted_index(index_file):
    """
    Loads an inverted index from a text file.
    Expected format per line:
        term df docid1,docid2,docid3,...
    Returns a dictionary mapping term -> set of document IDs (as integers).
    """
    inverted_index = {}
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            term = parts[0]
            posting_str = parts[2]
            doc_ids = posting_str.split(',')
            inverted_index[term] = set(int(doc_id) for doc_id in doc_ids if doc_id.isdigit())
    return inverted_index


def build_all_docs(inverted_index):
    """
    Constructs the universal set of document IDs from the inverted index.
    """
    all_docs = set()
    for postings in inverted_index.values():
        all_docs.update(postings)
    return all_docs


# -------------------------------------------------
# Part 2: Boolean Query Parsing and Evaluation
# -------------------------------------------------
def tokenize(query):
    """
    Tokenizes the query into terms, operators (AND, OR, NOT), and parentheses.
    Operators are standardized to uppercase and terms to lowercase.
    """
    # The regex captures words, parentheses, and the operators AND, OR, NOT.
    tokens = re.findall(r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|\w+', query, re.IGNORECASE)
    standardized_tokens = []
    for token in tokens:
        if token.upper() in ('AND', 'OR', 'NOT'):
            standardized_tokens.append(token.upper())
        elif token in ('(', ')'):
            standardized_tokens.append(token)
        else:
            standardized_tokens.append(token.lower())
    return standardized_tokens


class BooleanQueryParser:
    """
    A recursive descent parser for Boolean queries.
    Grammar (with precedence: NOT > AND > OR):
      Query     -> OrExpr
      OrExpr    -> AndExpr { OR AndExpr }*
      AndExpr   -> NotExpr { AND NotExpr }*
      NotExpr   -> [NOT] Primary
      Primary   -> term | '(' Query ')'

    The evaluation uses set operations on posting lists:
      - AND: intersection
      - OR: union
      - NOT: set difference with the universal set (all_docs)
    """

    def __init__(self, tokens, inverted_index, all_docs):
        self.tokens = tokens
        self.index = 0
        self.inverted_index = inverted_index
        self.all_docs = all_docs

    def current_token(self):
        if self.index < len(self.tokens):
            return self.tokens[self.index]
        return None

    def eat(self, token):
        if self.current_token() == token:
            self.index += 1
        else:
            raise Exception(f"Expected token '{token}', got '{self.current_token()}'")

    def parse_query(self):
        return self.parse_or()

    def parse_or(self):
        result = self.parse_and()
        while self.current_token() == 'OR':
            self.eat('OR')
            right = self.parse_and()
            result = result.union(right)
        return result

    def parse_and(self):
        result = self.parse_not()
        while self.current_token() == 'AND':
            self.eat('AND')
            right = self.parse_not()
            result = result.intersection(right)
        return result

    def parse_not(self):
        if self.current_token() == 'NOT':
            self.eat('NOT')
            operand = self.parse_not()
            return self.all_docs.difference(operand)
        else:
            return self.parse_primary()

    def parse_primary(self):
        token = self.current_token()
        if token == '(':
            self.eat('(')
            result = self.parse_query()
            self.eat(')')
            return result
        else:
            # Token is assumed to be a term.
            self.index += 1
            return self.inverted_index.get(token, set())


def process_boolean_query(query, inverted_index, all_docs):
    """
    Processes a Boolean query string and returns the set of document IDs
    that satisfy the query.
    """
    tokens = tokenize(query)
    parser = BooleanQueryParser(tokens, inverted_index, all_docs)
    result = parser.parse_query()
    return result


# -------------------------------------------------
# Part 3: Main Function – Sample Queries and Performance
# -------------------------------------------------
def main():
    # Define the path to the inverted index produced in Exercise 1.
    index_file = os.path.join("output", "inverted_index.txt")
    print("Loading inverted index...")
    inverted_index = load_inverted_index(index_file)
    all_docs = build_all_docs(inverted_index)
    print(f"Total documents in collection: {len(all_docs)}")

    # Define a diverse set of five Boolean queries.
    queries = [
        "real AND madrid",
        "real OR barcelona",
        "real AND NOT barcelona",
        "(champions AND real) OR barcelona",
        "liverpool OR (real AND NOT champions)"
    ]

    runtimes = []
    for query in queries:
        print(f"\nProcessing query: {query}")
        start_time = time.time()
        result = process_boolean_query(query, inverted_index, all_docs)
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        print(f"Query result (doc IDs): {sorted(result)}")
        print(f"Runtime: {elapsed:.6f} sec, {len(result)} documents retrieved")

    avg_runtime = sum(runtimes) / len(runtimes)
    print(f"\nAverage runtime over {len(queries)} queries: {avg_runtime:.6f} sec")


if __name__ == "__main__":
    main()
