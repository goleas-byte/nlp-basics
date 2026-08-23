import os
import re
import html
import string

import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "dataset"
)

IMDB_DATASET_PATH = os.path.join(
    DATASET_FOLDER,
    "IMDB Dataset.csv"
)

# Recommended sample size for the practical
RECOMMENDED_MIN_SAMPLES = 50

# Number of documents to display in detailed demonstrations
DISPLAY_DOCUMENTS = 3

# Number of documents for Task 8
TRANSFORMATION_DOCUMENTS = 5

# Maximum number of tokens displayed for long outputs
MAX_DISPLAY_TOKENS = 80


# ============================================================
# NLP OBJECTS
# ============================================================

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def print_success(message):
    print(f"\n[SUCCESS] {message}")


def print_warning(message):
    print(f"\n[WARNING] {message}")


def print_error(message):
    print(f"\n[ERROR] {message}")


# ============================================================
# NLTK RESOURCE SETUP
# ============================================================

def setup_nltk_resources():

    print_header("NLP RESOURCE CHECK")

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng",
         "averaged_perceptron_tagger_eng")
    ]

    all_available = True

    for resource_path, download_name in resources:

        try:
            nltk.data.find(resource_path)

            print(
                f"[SUCCESS] NLTK resource available: "
                f"{download_name}"
            )

        except LookupError:

            print(
                f"[WARNING] NLTK resource missing: "
                f"{download_name}"
            )

            print(
                f"Downloading {download_name}..."
            )

            try:
                success = nltk.download(
                    download_name,
                    quiet=False
                )

                if success:
                    print(
                        f"[SUCCESS] Downloaded: "
                        f"{download_name}"
                    )
                else:
                    print_error(
                        f"Unable to download: "
                        f"{download_name}"
                    )
                    all_available = False

            except Exception as error:

                print_error(
                    f"Error downloading {download_name}"
                )

                print(f"Details: {error}")

                all_available = False

    return all_available


# ============================================================
# STEP 1 - DATASET SELECTION AND VALIDATION
# ============================================================

def validate_dataset():

    print_header(
        "STEP 1 - DATASET SELECTION AND VALIDATION"
    )

    print(
        f"Project folder: {PROJECT_FOLDER}"
    )

    print(
        f"Dataset folder: {DATASET_FOLDER}"
    )

    print("\nSelected dataset:")

    print("IMDB Dataset.csv")

    if not os.path.exists(DATASET_FOLDER):

        print_error(
            "Dataset folder was not found."
        )

        return False

    if not os.path.exists(IMDB_DATASET_PATH):

        print_error(
            "IMDB Dataset.csv was not found "
            "inside the dataset folder."
        )

        print("\nAvailable files:")

        for file_name in os.listdir(DATASET_FOLDER):

            print(f" - {file_name}")

        return False

    print("\nDataset path found:")

    print(IMDB_DATASET_PATH)

    print_success(
        "STEP 1 completed successfully."
    )

    return True


# ============================================================
# STEP 2 - DATASET LOADING
# ============================================================

def load_dataset():

    print_header(
        "STEP 2 - DATASET LOADING"
    )

    try:

        print("Loading IMDB dataset...")

        df = pd.read_csv(
            IMDB_DATASET_PATH
        )

        required_columns = [
            "review",
            "sentiment"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            print_error(
                f"Required columns are missing: "
                f"{missing_columns}"
            )

            return None

        df["review"] = (
            df["review"]
            .fillna("")
            .astype(str)
        )

        df["sentiment"] = (
            df["sentiment"]
            .fillna("")
            .astype(str)
        )

        print_success(
            "Dataset loaded successfully."
        )

        print(
            f"\nTotal rows loaded: {len(df)}"
        )

        print(
            f"Total columns: {len(df.columns)}"
        )

        print("\nAvailable columns:")

        for column in df.columns:

            print(f" - {column}")

        return df

    except Exception as error:

        print_error(
            "Unable to load the dataset."
        )

        print(
            f"\nDetails: {error}"
        )

        return None


# ============================================================
# SAMPLE RANGE SELECTION
# ============================================================

def select_sample_range(df):

    print_header(
        "SAMPLE RANGE SELECTION"
    )

    total_samples = len(df)

    print(
        f"Total available samples: "
        f"{total_samples}"
    )

    print(
        "\nThe selected range will be used "
        "for detailed demonstrations."
    )

    print(
        "\nRecommended:"
    )

    print(
        "Select approximately 80-100 samples."
    )

    print(
        "Example: START = 101, END = 200"
    )

    while True:

        try:

            start = int(
                input(
                    f"\nEnter START sample number "
                    f"(1-{total_samples}): "
                )
            )

            end = int(
                input(
                    f"Enter END sample number "
                    f"(1-{total_samples}): "
                )
            )

            if start < 1:

                print_error(
                    "START must be at least 1."
                )

                continue

            if end > total_samples:

                print_error(
                    f"END cannot exceed "
                    f"{total_samples}."
                )

                continue

            if start > end:

                print_error(
                    "START cannot be greater "
                    "than END."
                )

                continue

            selected_count = (
                end - start + 1
            )

            if selected_count < RECOMMENDED_MIN_SAMPLES:

                print_warning(
                    f"You selected only "
                    f"{selected_count} samples."
                )

                print(
                    "At least 50 samples are "
                    "recommended for this practical."
                )

                choice = input(
                    "Continue? (y/n): "
                ).strip().lower()

                if choice != "y":
                    continue

            break

        except ValueError:

            print_error(
                "Please enter whole numbers only."
            )

    selected_df = df.iloc[
        start - 1:end
    ].copy()

    print(
        "\nSelected sample range:"
    )

    print(f"START: {start}")
    print(f"END:   {end}")

    print(
        f"\nNumber of selected samples: "
        f"{len(selected_df)}"
    )

    print_success(
        "Sample range selected successfully."
    )

    return selected_df, start, end


# ============================================================
# BASIC TOKENIZATION FOR STATISTICS
# ============================================================

def basic_tokens(text):

    return re.findall(
        r"\b[\w']+\b",
        text.lower()
    )


# ============================================================
# STEP 3 - DATASET EXPLORATION
# ============================================================

def explore_dataset(
    df,
    selected_df,
    start,
    end
):

    print_header(
        "STEP 3 - DATASET EXPLORATION"
    )

    # --------------------------------------------------------
    # COMPLETE DATASET
    # --------------------------------------------------------

    print("COMPLETE DATASET")

    print("-" * 70)

    print(
        f"Number of documents: {len(df)}"
    )

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
    )

    print(
        "\nSentiment distribution:"
    )

    for sentiment, count in sentiment_counts.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f" - {sentiment}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    lengths = [
        len(basic_tokens(text))
        for text in df["review"]
    ]

    average_length = (
        sum(lengths) / len(lengths)
        if lengths else 0
    )

    print(
        f"\nAverage document length: "
        f"{average_length:.2f} words"
    )

    print(
        "\nCalculating vocabulary..."
    )

    vocabulary = set()

    for text in df["review"]:

        vocabulary.update(
            basic_tokens(text)
        )

    print(
        "Number of unique words before "
        f"preprocessing: {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # SELECTED DATA
    # --------------------------------------------------------

    print_subheader(
        "SELECTED SAMPLE RANGE"
    )

    print(
        f"Selected range: "
        f"{start} - {end}"
    )

    print(
        f"Selected documents: "
        f"{len(selected_df)}"
    )

    selected_lengths = [
        len(basic_tokens(text))
        for text in selected_df["review"]
    ]

    selected_average = (
        sum(selected_lengths)
        / len(selected_lengths)
        if selected_lengths else 0
    )

    print(
        f"Average selected document length: "
        f"{selected_average:.2f} words"
    )

    selected_vocabulary = set()

    for text in selected_df["review"]:

        selected_vocabulary.update(
            basic_tokens(text)
        )

    print(
        f"Unique words in selected samples: "
        f"{len(selected_vocabulary)}"
    )

    # --------------------------------------------------------
    # RAW DOCUMENTS
    # --------------------------------------------------------

    print_subheader(
        "SAMPLE RAW DOCUMENTS"
    )

    for position in range(
        min(DISPLAY_DOCUMENTS, len(selected_df))
    ):

        row = selected_df.iloc[position]

        sample_number = (
            start + position
        )

        print(
            f"\n--- Sample {sample_number} ---"
        )

        print(
            f"Sentiment: "
            f"{row['sentiment']}"
        )

        print(
            "Raw review:"
        )

        preview = row["review"][:1000]

        if len(row["review"]) > 1000:
            preview += "..."

        print(preview)

    # --------------------------------------------------------
    # NOISE OBSERVATION
    # --------------------------------------------------------

    print_subheader(
        "MAJOR TYPES OF TEXTUAL NOISE OBSERVED"
    )

    print(
        "1. HTML tags such as <br />"
    )

    print(
        "2. Punctuation and repeated punctuation"
    )

    print(
        "3. Extra whitespace"
    )

    print(
        "4. Numbers and rating expressions"
    )

    print(
        "5. Special characters"
    )

    print(
        "6. Contractions such as don't, it's, "
        "and didn't"
    )

    print(
        "7. Capitalization differences"
    )

    print(
        "8. Informal movie-review expressions"
    )

    print_success(
        "STEP 3 completed successfully."
    )

    return {
        "full_vocabulary": vocabulary,
        "selected_vocabulary": selected_vocabulary,
        "average_length": average_length,
        "selected_average_length": selected_average
    }


# ============================================================
# STEP 4 - SENTENCE SEGMENTATION
# ============================================================

def segment_sentences(text):

    try:

        return sent_tokenize(text)

    except Exception:

        # Simple fallback if NLTK resource fails
        return re.split(
            r"(?<=[.!?])\s+",
            text
        )


def demonstrate_sentence_segmentation(
    selected_df,
    start
):

    print_header(
        "STEP 4 - SENTENCE SEGMENTATION"
    )

    print(
        "Sentence segmentation divides each "
        "document into individual sentences."
    )

    for position in range(
        min(3, len(selected_df))
    ):

        row = selected_df.iloc[position]

        sample_number = (
            start + position
        )

        sentences = segment_sentences(
            row["review"]
        )

        print(
            f"\n--- Sample {sample_number} ---"
        )

        print(
            f"Total sentences detected: "
            f"{len(sentences)}"
        )

        for number, sentence in enumerate(
            sentences[:10],
            start=1
        ):

            print(
                f"{number}. {sentence}"
            )

        if len(sentences) > 10:

            print(
                "... additional sentences omitted"
            )

    # --------------------------------------------------------
    # ABBREVIATION TEST
    # --------------------------------------------------------

    print_subheader(
        "DIFFICULT CASE: ABBREVIATION"
    )

    abbreviation_example = (
        "Dr. Sharma discussed NLP techniques. "
        "Students implemented them in Python."
    )

    print(
        "\nOriginal text:"
    )

    print(abbreviation_example)

    abbreviation_sentences = (
        segment_sentences(
            abbreviation_example
        )
    )

    print(
        "\nDetected sentences:"
    )

    for number, sentence in enumerate(
        abbreviation_sentences,
        start=1
    ):

        print(
            f"{number}. {sentence}"
        )

    print(
        "\nObservation:"
    )

    print(
        "Abbreviations such as Dr., Mr., "
        "U.S., etc. contain periods that "
        "do not necessarily indicate sentence "
        "boundaries."
    )

    print(
        "A sentence tokenizer must distinguish "
        "these abbreviation periods from actual "
        "sentence-ending punctuation."
    )

    print_success(
        "STEP 4 completed successfully."
    )

    return True


# ============================================================
# STEP 5 - TOKENIZATION COMPARISON
# ============================================================

def simple_split_tokenizer(text):

    return text.split()


def nltk_tokenizer(text):

    try:

        return word_tokenize(text)

    except Exception:

        return re.findall(
            r"\w+|[^\w\s]",
            text
        )


def regex_tokenizer(text):

    return re.findall(
        r"\w+(?:'\w+)?|[^\w\s]",
        text
    )


def display_tokens(tokens):

    if len(tokens) > MAX_DISPLAY_TOKENS:

        return (
            str(tokens[:MAX_DISPLAY_TOKENS])
            + " ..."
        )

    return str(tokens)


def compare_tokenizers(
    selected_df,
    start
):

    print_header(
        "STEP 5 - TOKENIZATION COMPARISON"
    )

    comparison_text = (
        "I really loved this movie! "
        "It cost $10, and I gave it 10/10. "
        "Email test@example.com."
    )

    print(
        "Demonstration text:"
    )

    print(comparison_text)

    split_tokens = (
        simple_split_tokenizer(
            comparison_text
        )
    )

    nltk_tokens = (
        nltk_tokenizer(
            comparison_text
        )
    )

    regex_tokens = (
        regex_tokenizer(
            comparison_text
        )
    )

    print_subheader(
        "TOKENIZATION RESULTS"
    )

    print(
        "\n1. Simple split()"
    )

    print(
        display_tokens(
            split_tokens
        )
    )

    print(
        "\n2. NLTK word_tokenize()"
    )

    print(
        display_tokens(
            nltk_tokens
        )
    )

    print(
        "\n3. Regular-expression tokenizer"
    )

    print(
        display_tokens(
            regex_tokens
        )
    )

    print_subheader(
        "THREE IMPORTANT DIFFERENCES"
    )

    print(
        "1. Punctuation:"
    )

    print(
        "split() usually keeps punctuation "
        "attached to words, while NLTK can "
        "produce punctuation as separate tokens."
    )

    print(
        "\n2. Contractions:"
    )

    print(
        "NLTK can separate apostrophe-based "
        "contractions differently from split()."
    )

    print(
        "\n3. Numbers and special expressions:"
    )

    print(
        "Expressions such as 10/10, $10 and "
        "email@example.com can be handled "
        "differently depending on the tokenizer."
    )

    # --------------------------------------------------------
    # Dataset example
    # --------------------------------------------------------

    print_subheader(
        "DATASET TOKENIZATION EXAMPLE"
    )

    if len(selected_df) > 0:

        sample_text = (
            selected_df.iloc[0]["review"]
        )

        print(
            f"\nSample {start}:"
        )

        print(
            "\nSimple split():"
        )

        print(
            display_tokens(
                simple_split_tokenizer(
                    sample_text
                )
            )
        )

        print(
            "\nNLTK tokenizer:"
        )

        print(
            display_tokens(
                nltk_tokenizer(
                    sample_text
                )
            )
        )

    print_success(
        "STEP 5 completed successfully."
    )

    return True


# ============================================================
# STEP 6 - TEXT NORMALIZATION AND CLEANING
# ============================================================

def lowercase_text(text):

    return text.lower()


def remove_html(text):

    text = html.unescape(text)

    return re.sub(
        r"<[^>]+>",
        " ",
        text
    )


def remove_urls(text):

    return re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )


def remove_emails(text):

    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )


def remove_numbers(text):

    return re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ",
        text
    )


def remove_punctuation(text):

    return text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


def normalize_whitespace(text):

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def clean_text(
    text,
    lowercase=True,
    remove_html_tags=True,
    remove_url=True,
    remove_email=True,
    remove_punct=True,
    remove_num=True,
    normalize_space=True
):

    if lowercase:

        text = lowercase_text(text)

    if remove_html_tags:

        text = remove_html(text)

    if remove_url:

        text = remove_urls(text)

    if remove_email:

        text = remove_emails(text)

    if remove_punct:

        text = remove_punctuation(text)

    if remove_num:

        text = remove_numbers(text)

    if normalize_space:

        text = normalize_whitespace(text)

    return text


def demonstrate_cleaning(
    selected_df,
    start
):

    print_header(
        "STEP 6 - TEXT NORMALIZATION AND CLEANING"
    )

    print(
        "The standard cleaning pipeline performs:"
    )

    print(
        "1. Lowercasing"
    )

    print(
        "2. HTML tag removal"
    )

    print(
        "3. URL removal"
    )

    print(
        "4. Email removal"
    )

    print(
        "5. Punctuation removal"
    )

    print(
        "6. Number removal"
    )

    print(
        "7. Whitespace normalization"
    )

    demonstration_text = (
        "<br /> I REALLY loved this movie! "
        "Visit https://example.com for more "
        "information. Email test@example.com. "
        "It received 10/10 ratings."
    )

    print_subheader(
        "CONTROLLED CLEANING DEMONSTRATION"
    )

    print(
        "\nOriginal:"
    )

    print(demonstration_text)

    text = lowercase_text(
        demonstration_text
    )

    print(
        "\nAfter lowercase:"
    )

    print(text)

    text = remove_html(text)

    print(
        "\nAfter HTML removal:"
    )

    print(text)

    text = remove_urls(text)

    print(
        "\nAfter URL removal:"
    )

    print(text)

    text = remove_emails(text)

    print(
        "\nAfter email removal:"
    )

    print(text)

    text = remove_punctuation(text)

    print(
        "\nAfter punctuation removal:"
    )

    print(text)

    text = remove_numbers(text)

    print(
        "\nAfter number removal:"
    )

    print(text)

    text = normalize_whitespace(text)

    print(
        "\nAfter whitespace normalization:"
    )

    print(text)

    print(
        "\nFinal cleaned text:"
    )

    print(
        clean_text(
            demonstration_text
        )
    )

    # --------------------------------------------------------
    # Dataset examples
    # --------------------------------------------------------

    print_subheader(
        "DATASET CLEANING DEMONSTRATION"
    )

    for position in range(
        min(3, len(selected_df))
    ):

        row = selected_df.iloc[position]

        sample_number = (
            start + position
        )

        original = row["review"]

        cleaned = clean_text(
            original
        )

        print(
            f"\nSAMPLE {sample_number}"
        )

        print(
            "\nBEFORE:"
        )

        print(
            original[:700]
        )

        if len(original) > 700:
            print("...")

        print(
            "\nAFTER:"
        )

        print(
            cleaned[:700]
        )

        if len(cleaned) > 700:
            print("...")

        print(
            f"\nCharacters before: "
            f"{len(original)}"
        )

        print(
            f"Characters after: "
            f"{len(cleaned)}"
        )

    print_success(
        "STEP 6 completed successfully."
    )

    return True


# ============================================================
# STOPWORD SETUP
# ============================================================

def get_stopwords():

    words = set(
        stopwords.words("english")
    )

    return words


# ============================================================
# STEP 7 - STOPWORD HANDLING
# ============================================================

def remove_stopwords(
    tokens,
    preserve_negation=True
):

    stop_words = get_stopwords()

    if preserve_negation:

        # Keep important sentiment words
        # because this is an IMDB sentiment dataset.
        stop_words -= {
            "not",
            "no",
            "nor",
            "never"
        }

    return [
        token
        for token in tokens
        if token.lower() not in stop_words
    ]


def demonstrate_stopword_handling():

    print_header(
        "STEP 7 - STOPWORD HANDLING"
    )

    example = (
        "This movie is not very good "
        "and I never recommend it."
    )

    tokens = nltk_tokenizer(
        example
    )

    print(
        "\nOriginal tokens:"
    )

    print(tokens)

    normal_result = remove_stopwords(
        tokens,
        preserve_negation=True
    )

    print(
        "\nAfter stopword removal "
        "with negation preservation:"
    )

    print(normal_result)

    aggressive_result = remove_stopwords(
        tokens,
        preserve_negation=False
    )

    print(
        "\nAfter aggressive stopword removal:"
    )

    print(aggressive_result)

    print(
        "\nObservation:"
    )

    print(
        "For movie sentiment analysis, words "
        "such as 'not', 'no' and 'never' can "
        "change the meaning of a sentence."
    )

    print(
        "Therefore the main IMDB pipeline "
        "preserves these negation words."
    )

    print_success(
        "STEP 7 completed successfully."
    )


# ============================================================
# POS CONVERSION FOR LEMMATIZATION
# ============================================================

def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return "a"

    if tag.startswith("V"):
        return "v"

    if tag.startswith("N"):
        return "n"

    if tag.startswith("R"):
        return "r"

    return "n"


def lemmatize_tokens(tokens):

    if not tokens:
        return []

    try:

        tagged_tokens = pos_tag(
            tokens
        )

    except Exception:

        tagged_tokens = [
            (token, "NN")
            for token in tokens
        ]

    result = []

    for token, tag in tagged_tokens:

        wordnet_pos = get_wordnet_pos(
            tag
        )

        lemma = lemmatizer.lemmatize(
            token,
            pos=wordnet_pos
        )

        result.append(
            lemma
        )

    return result


# ============================================================
# STEP 8 - STEMMING
# ============================================================

def stem_tokens(tokens):

    return [
        stemmer.stem(token)
        for token in tokens
    ]


def demonstrate_stemming():

    print_header(
        "STEP 8 - STEMMING"
    )

    examples = [
        "connect",
        "connected",
        "connecting",
        "connection",
        "study",
        "studies",
        "studying",
        "studied",
        "compute",
        "computer",
        "computing",
        "computation"
    ]

    print(
        "\nPorter Stemmer results:"
    )

    print(
        f"{'Original Word':<20}"
        f"{'Stemmed Word':<20}"
    )

    print(
        "-" * 40
    )

    for word in examples:

        stem = stemmer.stem(
            word
        )

        print(
            f"{word:<20}"
            f"{stem:<20}"
        )

    print(
        "\nObservation:"
    )

    print(
        "Stemming is rule-based and may "
        "produce truncated or linguistically "
        "unnatural forms."
    )

    print_success(
        "STEP 8 completed successfully."
    )


# ============================================================
# STEP 9 - LEMMATIZATION
# ============================================================

def demonstrate_lemmatization():

    print_header(
        "STEP 9 - LEMMATIZATION"
    )

    examples = [
        "studies",
        "running",
        "better",
        "computers",
        "connected",
        "children",
        "mice",
        "cars"
    ]

    tokens = examples

    stems = stem_tokens(
        tokens
    )

    lemmas = lemmatize_tokens(
        tokens
    )

    print(
        f"\n{'Original':<20}"
        f"{'Stemmed':<20}"
        f"{'Lemmatized':<20}"
    )

    print(
        "-" * 60
    )

    for original, stem, lemma in zip(
        tokens,
        stems,
        lemmas
    ):

        print(
            f"{original:<20}"
            f"{stem:<20}"
            f"{lemma:<20}"
        )

    print_subheader(
        "STEMMING VS LEMMATIZATION"
    )

    print(
        "Stemming removes or modifies word "
        "endings using heuristic rules."
    )

    print(
        "Lemmatization attempts to return the "
        "dictionary/base form of the word."
    )

    print(
        "POS information is used so that "
        "words such as running can be reduced "
        "toward their correct base form."
    )

    print_success(
        "STEP 9 completed successfully."
    )


# ============================================================
# PIPELINE TOKEN PROCESSING
# ============================================================

def tokenize_clean_text(text):

    return nltk_tokenizer(
        text
    )


def filter_word_tokens(tokens):

    return [
        token
        for token in tokens
        if re.search(
            r"[A-Za-z]",
            token
        )
    ]


# ============================================================
# COMPLETE REUSABLE PIPELINE
# ============================================================

def preprocess_text(
    text,
    do_sentence_segmentation=True,
    do_tokenization=True,
    do_cleaning=True,
    do_normalization=True,
    do_stopwords=True,
    do_stemming=True,
    do_lemmatization=False,
    preserve_negation=True,
    remove_html_tags=True,
    remove_url=True,
    remove_email=True,
    remove_punct=True,
    remove_num=True,
    normalize_space=True
):
    """
    Reusable preprocessing function.

    Major stages can be independently enabled/disabled:
    sentence segmentation, tokenization, cleaning, normalization,
    stopword handling, stemming and lemmatization.

    Individual cleaning operations can also be enabled/disabled.
    """
    if do_sentence_segmentation:
        sentences = segment_sentences(text)
    else:
        sentences = [text]

    all_tokens = []

    for sentence in sentences:
        current_text = sentence

        if do_cleaning:
            current_text = clean_text(
                current_text,
                lowercase=do_normalization,
                remove_html_tags=remove_html_tags,
                remove_url=remove_url,
                remove_email=remove_email,
                remove_punct=remove_punct,
                remove_num=remove_num,
                normalize_space=normalize_space
            )
        elif do_normalization:
            current_text = current_text.lower()
            if normalize_space:
                current_text = normalize_whitespace(current_text)

        if do_tokenization:
            tokens = nltk_tokenizer(current_text)
            tokens = filter_word_tokens(tokens)
        else:
            # Tokenization is genuinely disabled: retain the current
            # sentence as one text item instead of tokenizing it.
            tokens = [current_text] if current_text.strip() else []

        if do_stopwords:
            tokens = remove_stopwords(
                tokens,
                preserve_negation=preserve_negation
            )

        all_tokens.extend(tokens)

    stemmed = stem_tokens(all_tokens) if do_stemming else None
    lemmatized = (
        lemmatize_tokens(all_tokens)
        if do_lemmatization else None
    )

    return {
        "tokens": all_tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    }


def demonstrate_pipeline_configuration():
    """Verify that major preprocessing operations can be toggled."""
    print_header("PIPELINE CONFIGURATION TEST")

    sample = (
        "This is NOT a useful example! "
        "Visit https://example.com and call 123."
    )

    print(
        "preprocess_text() supports independent enable/disable controls "
        "for the major preprocessing stages."
    )

    print("\nConfiguration A - full preprocessing:")
    print(preprocess_text(
        sample,
        do_sentence_segmentation=True,
        do_tokenization=True,
        do_cleaning=True,
        do_normalization=True,
        do_stopwords=True,
        do_stemming=True,
        do_lemmatization=True,
        preserve_negation=True
    ))

    print("\nConfiguration B - cleaning, stopwords and stemming disabled:")
    print(preprocess_text(
        sample,
        do_sentence_segmentation=True,
        do_tokenization=True,
        do_cleaning=False,
        do_normalization=False,
        do_stopwords=False,
        do_stemming=False,
        do_lemmatization=False
    ))

    print("\nConfiguration C - keep numbers while cleaning other noise:")
    print(preprocess_text(
        sample,
        do_sentence_segmentation=True,
        do_tokenization=True,
        do_cleaning=True,
        do_normalization=True,
        do_stopwords=True,
        do_stemming=False,
        do_lemmatization=False,
        preserve_negation=True,
        remove_num=False
    ))

    print(
        "\nObservation: major stages and individual cleaning operations "
        "can be enabled or disabled without rewriting the pipeline."
    )
    print_success("Pipeline configuration controls verified.")


# ============================================================
# STEP 10 - COMPLETE PREPROCESSING PIPELINE
# ============================================================

def demonstrate_complete_pipeline(
    selected_df
):

    print_header(
        "STEP 10 - COMPLETE PREPROCESSING PIPELINE"
    )

    print(
        "Pipeline:"
    )

    print(
        "Raw Text"
    )

    print(
        "   ↓"
    )

    print(
        "Sentence Segmentation"
    )

    print(
        "   ↓"
    )

    print(
        "Tokenization"
    )

    print(
        "   ↓"
    )

    print(
        "Cleaning"
    )

    print(
        "   ↓"
    )

    print(
        "Normalization"
    )

    print(
        "   ↓"
    )

    print(
        "Stopword Handling"
    )

    print(
        "   ↓"
    )

    print(
        "Stemming + Lemmatization"
    )

    print(
        "   ↓"
    )

    print(
        "Final Preprocessed Representation"
    )

    if len(selected_df) == 0:

        return

    sample_text = (
        selected_df.iloc[0]["review"]
    )

    result = preprocess_text(
        sample_text,
        do_sentence_segmentation=True,
        do_tokenization=True,
        do_cleaning=True,
        do_normalization=True,
        do_stopwords=True,
        do_stemming=True,
        do_lemmatization=True,
        preserve_negation=True
    )

    print_subheader(
        "PIPELINE OUTPUT FOR ONE DOCUMENT"
    )

    print(
        "\nOriginal text:"
    )

    print(
        sample_text[:1000]
    )

    print(
        "\nFinal tokens:"
    )

    print(
        display_tokens(
            result["tokens"]
        )
    )

    print(
        "\nStemmed tokens:"
    )

    print(
        display_tokens(
            result["stemmed"]
        )
    )

    print(
        "\nLemmatized tokens:"
    )

    print(
        display_tokens(
            result["lemmatized"]
        )
    )

    print_success(
        "STEP 10 completed successfully."
    )


# ============================================================
# STEP 11 - PROCESS SELECTED DATASET SAMPLES
# ============================================================

def process_selected_samples(
    selected_df,
    start
):

    print_header(
        "STEP 11 - PROCESS SELECTED DATASET SAMPLES"
    )

    processed_rows = []

    for position in range(
        len(selected_df)
    ):

        row = selected_df.iloc[position]

        result = preprocess_text(
            row["review"],
            do_sentence_segmentation=True,
            do_tokenization=True,
            do_cleaning=True,
            do_normalization=True,
            do_stopwords=True,
            do_stemming=True,
            do_lemmatization=True,
            preserve_negation=True
        )

        processed_rows.append(
            {
                "sample_number":
                    start + position,

                "sentiment":
                    row["sentiment"],

                "original":
                    row["review"],

                "tokens":
                    result["tokens"],

                "stemmed":
                    result["stemmed"],

                "lemmatized":
                    result["lemmatized"]
            }
        )

    print(
        f"Documents processed: "
        f"{len(processed_rows)}"
    )

    print_success(
        "STEP 11 completed successfully."
    )

    return processed_rows


# ============================================================
# STEP 12 - BEFORE VS AFTER STATISTICS
# ============================================================

def calculate_before_statistics(
    df
):

    total_tokens = 0

    vocabulary = set()

    for text in df["review"]:

        tokens = basic_tokens(
            text
        )

        total_tokens += len(tokens)

        vocabulary.update(
            tokens
        )

    average = (
        total_tokens / len(df)
        if len(df) > 0
        else 0
    )

    return {
        "total_tokens":
            total_tokens,

        "unique_tokens":
            len(vocabulary),

        "average_tokens":
            average,

        "vocabulary_size":
            len(vocabulary)
    }


def calculate_after_statistics(
    df
):

    total_tokens = 0

    vocabulary = set()

    for index, text in enumerate(
        df["review"],
        start=1
    ):

        result = preprocess_text(
            text,
            do_sentence_segmentation=False,
            do_tokenization=True,
            do_cleaning=True,
            do_normalization=True,
            do_stopwords=True,
            do_stemming=True,
            do_lemmatization=False,
            preserve_negation=True
        )

        tokens = result["stemmed"]

        if tokens is None:
            tokens = []

        total_tokens += len(tokens)

        vocabulary.update(
            tokens
        )

        if index % 5000 == 0:

            print(
                f"Processed "
                f"{index}/{len(df)} documents..."
            )

    average = (
        total_tokens / len(df)
        if len(df) > 0
        else 0
    )

    return {
        "total_tokens":
            total_tokens,

        "unique_tokens":
            len(vocabulary),

        "average_tokens":
            average,

        "vocabulary_size":
            len(vocabulary)
    }


def display_statistics_table(
    before,
    after
):

    print_header(
        "STEP 12 - BEFORE VS AFTER ANALYSIS"
    )

    print(
        f"{'Measure':<30}"
        f"{'Before Preprocessing':>22}"
        f"{'After Preprocessing':>22}"
    )

    print(
        "-" * 74
    )

    print(
        f"{'Total tokens':<30}"
        f"{before['total_tokens']:>22}"
        f"{after['total_tokens']:>22}"
    )

    print(
        f"{'Unique tokens':<30}"
        f"{before['unique_tokens']:>22}"
        f"{after['unique_tokens']:>22}"
    )

    print(
        f"{'Average tokens/document':<30}"
        f"{before['average_tokens']:>22.2f}"
        f"{after['average_tokens']:>22.2f}"
    )

    print(
        f"{'Vocabulary size':<30}"
        f"{before['vocabulary_size']:>22}"
        f"{after['vocabulary_size']:>22}"
    )

    reduction = (
        (
            before["vocabulary_size"]
            - after["vocabulary_size"]
        )
        / before["vocabulary_size"]
        * 100
    )

    print(
        f"\nVocabulary reduction: "
        f"{reduction:.2f}%"
    )

    print(
        "\nExplanation:"
    )

    print("Correct interpretation:")
    print(
        "The total token count and average tokens per document decrease "
        "because cleaning, stopword removal and normalization remove or "
        "consolidate text."
    )
    print(
        "Vocabulary size does not have to decrease. In this implementation "
        "the BEFORE vocabulary is built from basic raw-text tokens, while "
        "the AFTER vocabulary is built from cleaned, stopword-filtered and "
        "Porter-stemmed tokens. These are different token representations."
    )
    print(
        "Therefore, the observed vocabulary may decrease or increase. "
        "An increase does not mean preprocessing failed; it means the "
        "before/after vocabularies are not directly identical token spaces."
    )
    print(
        "Cleaning can split HTML fragments and normalize attached text, "
        "while stemming can create stem forms that were not present as "
        "exact raw tokens. Vocabulary should therefore be reported as a "
        "change rather than automatically called a reduction."
    )

    print_success(
        "STEP 12 completed successfully."
    )


# ============================================================
# STEP 13 - SAMPLE TRANSFORMATION
# ============================================================

def demonstrate_five_transformations(
    df
):

    print_header(
        "STEP 13 - SAMPLE TRANSFORMATION DEMONSTRATION"
    )

    print(
        "Five documents are transformed through:"
    )

    print(
        "Original → Tokenized → Normalized "
        "→ Stemmed → Lemmatized"
    )

    number_of_documents = min(
        TRANSFORMATION_DOCUMENTS,
        len(df)
    )

    for position in range(
        number_of_documents
    ):

        text = df.iloc[
            position
        ]["review"]

        print_subheader(
            f"DOCUMENT {position + 1}"
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        print(
            "\nORIGINAL TEXT:"
        )

        print(
            text[:800]
        )

        if len(text) > 800:
            print("...")

        # ----------------------------------------------------
        # Tokenized
        # ----------------------------------------------------

        tokens = nltk_tokenizer(
            text
        )

        print(
            "\nTOKENIZED TEXT:"
        )

        print(
            display_tokens(
                tokens
            )
        )

        # ----------------------------------------------------
        # Normalized
        # ----------------------------------------------------

        normalized = clean_text(
            text,
            lowercase=True,
            remove_html_tags=True,
            remove_url=True,
            remove_email=True,
            remove_punct=True,
            remove_num=True,
            normalize_space=True
        )

        print(
            "\nNORMALIZED TEXT:"
        )

        print(
            normalized[:800]
        )

        if len(normalized) > 800:
            print("...")

        # ----------------------------------------------------
        # Stemmed
        # ----------------------------------------------------

        normalized_tokens = filter_word_tokens(
            nltk_tokenizer(
                normalized
            )
        )

        normalized_tokens = remove_stopwords(
            normalized_tokens,
            preserve_negation=True
        )

        stems = stem_tokens(
            normalized_tokens
        )

        print(
            "\nSTEMMED TEXT:"
        )

        print(
            display_tokens(
                stems
            )
        )

        # ----------------------------------------------------
        # Lemmatized
        # ----------------------------------------------------

        lemmas = lemmatize_tokens(
            normalized_tokens
        )

        print(
            "\nLEMMATIZED TEXT:"
        )

        print(
            display_tokens(
                lemmas
            )
        )

    print_success(
        "STEP 13 completed successfully."
    )


# ============================================================
# STEP 14 - AGGRESSIVE VS MEANING-PRESERVING
# ============================================================

def aggressive_clinical_pipeline(
    text
):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = remove_punctuation(
        text
    )

    # Remove numbers
    text = remove_numbers(
        text
    )

    # Tokenize
    tokens = nltk_tokenizer(
        text
    )

    # Remove all stopwords
    tokens = remove_stopwords(
        tokens,
        preserve_negation=False
    )

    # Stem
    tokens = stem_tokens(
        tokens
    )

    return tokens


def meaning_preserving_clinical_pipeline(
    text
):

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = normalize_whitespace(
        text
    )

    # Keep important clinical information:
    # - negation
    # - numbers
    # - units
    # - slash in mg/dL

    tokens = nltk_tokenizer(
        text
    )

    # Remove only obvious punctuation
    # while preserving meaningful clinical forms.
    meaningful_tokens = []

    for token in tokens:

        if re.match(
            r"^\d+(?:\.\d+)?$",
            token
        ):

            meaningful_tokens.append(
                token
            )

        elif re.match(
            r"^\d+(?:\.\d+)?/[A-Za-z]+$",
            token
        ):

            meaningful_tokens.append(
                token
            )

        elif re.search(
            r"[A-Za-z]",
            token
        ):

            cleaned = re.sub(
                r"^[^\w]+|[^\w]+$",
                "",
                token
            )

            if cleaned:

                meaningful_tokens.append(
                    cleaned
                )

    # Remove only unimportant stopwords,
    # preserving clinical negation.
    stop_words = get_stopwords()

    stop_words -= {
        "not",
        "no",
        "nor",
        "never"
    }

    final_tokens = [
        token
        for token in meaningful_tokens
        if token.lower()
        not in stop_words
        or token.lower()
        in {"not", "no", "nor", "never"}
    ]

    return final_tokens


def demonstrate_clinical_pipelines():

    print_header(
        "STEP 14 - AGGRESSIVE VS MEANING-PRESERVING PIPELINE"
    )

    clinical_sentence = (
        "The patient is not suffering from diabetes, "
        "but his glucose level increased to 180 mg/dL."
    )

    print(
        "\nClinical sentence:"
    )

    print(
        clinical_sentence
    )

    aggressive = aggressive_clinical_pipeline(
        clinical_sentence
    )

    meaning_preserving = (
        meaning_preserving_clinical_pipeline(
            clinical_sentence
        )
    )

    print_subheader(
        "PIPELINE A - AGGRESSIVE PREPROCESSING"
    )

    print(
        "Operations:"
    )

    print(
        "- Remove stopwords"
    )

    print(
        "- Remove punctuation"
    )

    print(
        "- Remove numbers"
    )

    print(
        "- Apply stemming"
    )

    print(
        "\nOutput:"
    )

    print(
        aggressive
    )

    print_subheader(
        "PIPELINE B - MEANING-PRESERVING PREPROCESSING"
    )

    print(
        "Important clinical information is preserved:"
    )

    print(
        "- Negation"
    )

    print(
        "- Numeric glucose value"
    )

    print(
        "- Clinical unit"
    )

    print(
        "- Important medical terminology"
    )

    print(
        "\nOutput:"
    )

    print(
        meaning_preserving
    )

    print_subheader(
        "CRITICAL ANALYSIS"
    )

    print(
        "Important information lost by Pipeline A:"
    )

    print(
        "1. The word 'not' can be removed, "
        "changing the meaning from absence of "
        "diabetes to possible diabetes."
    )

    print(
        "2. The number 180 is removed, losing "
        "the actual glucose measurement."
    )

    print(
        "3. The unit mg/dL may be damaged or "
        "removed, making the numerical value "
        "less clinically meaningful."
    )

    print(
        "4. Stemming can produce unnatural "
        "word forms."
    )

    print(
        "\nWhy blind preprocessing can reduce "
        "NLP quality:"
    )

    print(
        "Traditional preprocessing assumes that "
        "all punctuation, stopwords and numbers "
        "are noise. In clinical text, however, "
        "negation, numerical measurements, units "
        "and terminology can carry the central "
        "meaning of the sentence."
    )

    print_success(
        "STEP 14 completed successfully."
    )


# ============================================================
# DATASET STEMMING VS LEMMATIZATION EXAMPLES
# ============================================================

def find_dataset_stem_lemma_examples(
    selected_df
):

    print_header(
        "STEMMING VS LEMMATIZATION - DATASET EXAMPLES"
    )

    found = []

    for text in selected_df["review"]:

        cleaned = clean_text(
            text
        )

        tokens = filter_word_tokens(
            nltk_tokenizer(
                cleaned
            )
        )

        tokens = remove_stopwords(
            tokens,
            preserve_negation=True
        )

        stems = stem_tokens(
            tokens
        )

        lemmas = lemmatize_tokens(
            tokens
        )

        for original, stem, lemma in zip(
            tokens,
            stems,
            lemmas
        ):

            if (
                stem != lemma
                and len(original) > 3
                and len(stem) > 2
            ):

                example = (
                    original,
                    stem,
                    lemma
                )

                if example not in found:

                    found.append(
                        example
                    )

            if len(found) >= 10:
                break

        if len(found) >= 10:
            break

    print(
        f"{'Original':<20}"
        f"{'Stemmed':<20}"
        f"{'Lemmatized':<20}"
    )

    print(
        "-" * 60
    )

    for original, stem, lemma in found:

        print(
            f"{original:<20}"
            f"{stem:<20}"
            f"{lemma:<20}"
        )

    print(
        "\nAt least five examples are displayed "
        "where stemming and lemmatization behave "
        "differently."
    )


# ============================================================
# STEP 15 - FINAL SUMMARY
# ============================================================

def final_summary():

    print_header(
        "STEP 15 - FINAL ANALYSIS"
    )

    print(
        "The NLP preprocessing pipeline has "
        "implemented the required stages."
    )

    print(
        "\nCompleted stages:"
    )

    stages = [
        "Dataset selection and validation",
        "Dataset loading",
        "Dataset exploration",
        "Sentence segmentation",
        "Tokenization comparison",
        "Text normalization and cleaning",
        "Stopword handling",
        "Stemming",
        "Lemmatization",
        "Complete preprocessing pipeline with configurable operations",
        "Selected dataset processing",
        "Before vs after statistics",
        "Five-document transformation",
        "Aggressive vs meaning-preserving analysis"
    ]

    for number, stage in enumerate(
        stages,
        start=1
    ):

        print(
            f"[SUCCESS] {stage}"
        )

    print(
        "\nThe final representation is suitable "
        "for subsequent NLP tasks such as "
        "classification, sentiment analysis, "
        "feature extraction and machine learning."
    )

    print_success(
        "NLP preprocessing practical completed."
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print_header(
        "NLP TEXT PREPROCESSING PIPELINE"
    )

    print(
        "Starting the complete practical implementation..."
    )

    # --------------------------------------------------------
    # NLP resources
    # --------------------------------------------------------

    if not setup_nltk_resources():

        print_error(
            "Required NLTK resources are unavailable."
        )

        return

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    if not validate_dataset():

        print_error(
            "Program stopped because dataset "
            "validation failed."
        )

        return

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    df = load_dataset()

    if df is None:

        print_error(
            "Program stopped because dataset "
            "loading failed."
        )

        return

    # --------------------------------------------------------
    # Sample selection
    # --------------------------------------------------------

    selected_df, start, end = (
        select_sample_range(df)
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    explore_dataset(
        df,
        selected_df,
        start,
        end
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    demonstrate_sentence_segmentation(
        selected_df,
        start
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    compare_tokenizers(
        selected_df,
        start
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    demonstrate_cleaning(
        selected_df,
        start
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    demonstrate_stopword_handling()

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    demonstrate_stemming()

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    demonstrate_lemmatization()

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    demonstrate_complete_pipeline(
        selected_df
    )

    # Explicitly demonstrate the assignment requirement that major
    # preprocessing operations can be enabled or disabled.
    demonstrate_pipeline_configuration()

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    processed_samples = (
        process_selected_samples(
            selected_df,
            start
        )
    )

    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    print(
        "\nCalculating BEFORE preprocessing "
        "statistics for complete dataset..."
    )

    before_stats = (
        calculate_before_statistics(
            df
        )
    )

    print(
        "\nCalculating AFTER preprocessing "
        "statistics for complete dataset..."
    )

    after_stats = (
        calculate_after_statistics(
            df
        )
    )

    display_statistics_table(
        before_stats,
        after_stats
    )

    # --------------------------------------------------------
    # STEP 13
    # --------------------------------------------------------

    demonstrate_five_transformations(
        df
    )

    # --------------------------------------------------------
    # Dataset stem vs lemma examples
    # --------------------------------------------------------

    find_dataset_stem_lemma_examples(
        selected_df
    )

    # --------------------------------------------------------
    # STEP 14
    # --------------------------------------------------------

    demonstrate_clinical_pipelines()

    # --------------------------------------------------------
    # STEP 15
    # --------------------------------------------------------

    final_summary()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()