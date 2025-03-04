#!/usr/bin/env python3
import os
import string
import time
import itertools


# ---------------------------
# Part 1: Load the Inverted Index
# ---------------------------
def load_inverted_index(index_file):
    """
    Load the inverted index from a text file.
    Expected format for each line:
        term df docid1,docid2,docid3,...
    Returns a dictionary mapping each term (string) to a set of document IDs (integers).
    """
    inverted_index = {}
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip and split the line by whitespace (first two parts are term and df)
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            term = parts[0]
            # parts[1] is the document frequency, which we can ignore because we'll get it from the list
            posting_str = parts[2]
            # If there is a comma-separated list of docIDs (there might be no spaces)
            doc_ids = posting_str.split(',')
            # Convert doc IDs to integers
            inverted_index[term] = set(int(doc_id) for doc_id in doc_ids if doc_id.isdigit())
    return inverted_index


# ---------------------------
# Part 2: Create 26 Letter Lists from the Inverted Index
# ---------------------------
def create_letter_lists(inverted_index):
    """
    From an inverted index (term -> posting list), produce 26 lists
    mapping each letter (a-z) to the set of document IDs that contain
    at least one term starting with that letter.
    """
    letter_lists = {letter: set() for letter in string.ascii_lowercase}

    for term, doc_ids in inverted_index.items():
        if term:  # Ensure term is not empty
            first_char = term[0].lower()
            if first_char in letter_lists:
                letter_lists[first_char].update(doc_ids)

    # Convert each set to a sorted list (for consistent ordering in intersections)
    for letter in letter_lists:
        letter_lists[letter] = sorted(letter_lists[letter])

    return letter_lists


def save_letter_lists(letter_lists, output_file):
    """
    Save all 26 letter lists to a single file.
    Each line contains a letter, a TAB, and then a space-separated list of document IDs.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for letter in sorted(letter_lists.keys()):
            doc_ids_str = " ".join(str(doc_id) for doc_id in letter_lists[letter])
            f.write(f"{letter}\t{doc_ids_str}\n")


# ---------------------------
# Part 3: Intersection Algorithms
# ---------------------------
def intersect_linear(list1, list2):
    """
    Intersection using a two-pointer (merge-based) approach.
    Assumes list1 and list2 are sorted lists of document IDs.
    Returns a list of common document IDs.
    """
    i, j = 0, 0
    result = []
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    return result


def binary_search(arr, target, low, high):
    """
    Binary search for target in arr between indices low and high.
    Returns the index of target if found; otherwise returns -1.
    """
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1


def intersect_galloping(list1, list2):
    """
    Intersection using a galloping (exponential search) algorithm.
    For each element in the smaller list, performs exponential search in the larger list.
    Assumes both lists are sorted.
    """
    # Ensure list1 is the smaller list
    if len(list1) > len(list2):
        list1, list2 = list2, list1

    result = []
    for element in list1:
        lo = 0
        hi = 1
        # Exponentially increase hi until we pass element or reach the end of list2.
        while hi < len(list2) and list2[hi] < element:
            lo = hi
            hi *= 2
        hi = min(hi, len(list2))
        pos = binary_search(list2, element, lo, hi)
        if pos != -1:
            result.append(element)
    return result


# ---------------------------
# Part 4: Measure Intersection Algorithms
# ---------------------------
def measure_intersection_algorithms(letter_lists):
    """
    For all unique pairs of letter lists (325 pairs for letters a-z),
    intersect them using both the linear and galloping algorithms.
    Measures and returns the total running time and rate (elements per second)
    for each algorithm.
    """
    total_time_linear = 0.0
    total_time_galloping = 0.0
    total_elements_processed = 0  # Sum of lengths of each list pair

    letters = sorted(letter_lists.keys())
    for letter1, letter2 in itertools.combinations(letters, 2):
        list1 = letter_lists[letter1]
        list2 = letter_lists[letter2]

        # Count total elements in the pair (for later rate calculation)
        total_elements_processed += (len(list1) + len(list2))

        # Time the linear intersection
        start = time.time()
        result_linear = intersect_linear(list1, list2)
        total_time_linear += time.time() - start

        # Time the galloping intersection
        start = time.time()
        result_galloping = intersect_galloping(list1, list2)
        total_time_galloping += time.time() - start

        # Verify both algorithms yield the same result (order does not matter)
        if sorted(result_linear) != sorted(result_galloping):
            print(f"Mismatch for letters {letter1} and {letter2}!")

    rate_linear = total_elements_processed / total_time_linear if total_time_linear > 0 else 0
    rate_galloping = total_elements_processed / total_time_galloping if total_time_galloping > 0 else 0

    return {
        "rate_linear": rate_linear,
        "rate_galloping": rate_galloping,
        "total_elements": total_elements_processed,
        "total_time_linear": total_time_linear,
        "total_time_galloping": total_time_galloping
    }


# ---------------------------
# Part 5: Measure Disk Read Rate for the 26 Lists
# ---------------------------
def measure_disk_read_rate(letter_lists_file):
    """
    Measure the rate of reading the 26 lists from disk.
    Each line in the file is assumed to be in the format:
       letter<TAB>docid1 docid2 docid3 ...
    Returns the read rate in elements per second, along with total elements and time.
    """
    start = time.time()
    total_elements = 0
    with open(letter_lists_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            # Count the number of document IDs (split by spaces)
            doc_ids = parts[1].split()
            total_elements += len(doc_ids)
    total_time = time.time() - start
    rate = total_elements / total_time if total_time > 0 else 0
    return rate, total_elements, total_time


# ---------------------------
# Main Function to Tie Everything Together
# ---------------------------
def main():
    # Define file paths
    index_file = os.path.join("output", "inverted_index.txt")
    letter_lists_file = os.path.join("output", "letter_lists.txt")

    # --- Step 1: Load the Inverted Index ---
    print("Loading inverted index...")
    inverted_index = load_inverted_index(index_file)

    # --- Step 2: Create 26 Letter Lists ---
    print("Creating letter lists from the inverted index...")
    letter_lists = create_letter_lists(inverted_index)

    # Save the letter lists to a file (all in one file)
    print(f"Saving letter lists to {letter_lists_file}...")
    os.makedirs(os.path.dirname(letter_lists_file), exist_ok=True)
    save_letter_lists(letter_lists, letter_lists_file)

    # --- Step 3: Measure Intersection Algorithms ---
    print("Measuring intersection algorithms over all letter list pairs...")
    intersection_stats = measure_intersection_algorithms(letter_lists)
    print(f"Total elements processed (over all pairs): {intersection_stats['total_elements']}")
    print(f"Linear Intersection Time: {intersection_stats['total_time_linear']:.6f} sec")
    print(f"Galloping Intersection Time: {intersection_stats['total_time_galloping']:.6f} sec")
    print(f"Linear Intersection Rate: {intersection_stats['rate_linear']:.2f} elements/sec")
    print(f"Galloping Intersection Rate: {intersection_stats['rate_galloping']:.2f} elements/sec")

    # --- Step 4: Measure Disk Read Rate for the Letter Lists ---
    print("Measuring disk read rate for the letter lists file...")
    read_rate, total_read_elements, read_time = measure_disk_read_rate(letter_lists_file)
    print(f"Disk Read Rate: {read_rate:.2f} elements/sec (Read {total_read_elements} elements in {read_time:.6f} sec)")

    # Optionally, print machine information (to include on your slide)
    try:
        import platform
        print("\nMachine and Environment Details:")
        print(f"Programming Language: Python {platform.python_version()}")
        print(f"Machine: {platform.machine()} on {platform.system()} {platform.release()}")
    except Exception as e:
        print("Unable to get machine information:", e)


if __name__ == "__main__":
    main()
