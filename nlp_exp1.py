
"""
NLP EXPERIMENT 1
Implementation and Evaluation of N-Gram Language Models

Dataset directory:
C:\\Users\\golea\\Desktop\\Mtech_sem_1\\nlp-basics\\mobiletext

Python 3.11+
External libraries: None
"""

import re
import math
import random
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mobiletext"

# Large corpus files are read in chunks up to these limits.
# Increase these values if your computer has sufficient RAM.
MAX_TRAIN_CHARS = 5_000_000
MAX_TEST_CHARS = 1_000_000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

START = "<s>"
END = "</s>"
UNK = "<unk>"


# ============================================================
# 1. CORPUS DISCOVERY AND LOADING
# ============================================================

def discover_files():
    """Find text datasets in mobiletext and its subfolders."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_DIR}"
        )

    files = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            # Accept extensionless files and common text formats.
            if path.suffix.lower() in ("", ".txt", ".text", ".csv"):
                files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def choose_file(files, title):
    """Display datasets and let the user select one."""
    if not files:
        return None

    print(f"\n{title}")
    print("-" * 75)

    for i, path in enumerate(files, 1):
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0

        relative = path.relative_to(DATA_DIR)
        print(f"{i:3}. {relative} ({size_mb:.2f} MB)")

    while True:
        choice = input(
            "\nEnter file number, or 0 to return: "
        ).strip()

        if choice == "0":
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]

        print("Invalid choice. Please enter a listed number.")


def read_text_file(path, max_chars):
    """
    Read a text file safely.

    For extremely large datasets, only the first max_chars
    characters are loaded to avoid excessive memory usage.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding,
                      errors="strict") as file:
                text = file.read(max_chars)
            return text
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8",
              errors="replace") as file:
        return file.read(max_chars)


# ============================================================
# 2. PREPROCESSING AND TOKENIZATION
# ============================================================

def clean_text(text):
    """Normalize text and remove unnecessary symbols."""
    text = text.lower()

    # Keep letters, digits, whitespace and sentence punctuation.
    text = re.sub(r"[^a-z0-9\s.!?'’-]", " ", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_sentences(text):
    """
    Split a corpus into sentences and tokenize each sentence.

    Newlines are treated as possible message boundaries before
    normalization, which is useful for message datasets.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)
        raw_sentences.extend(part for part in parts if part.strip())

    sentences = []

    for sentence in raw_sentences:
        normalized = clean_text(sentence)
        tokens = re.findall(
            r"[a-z0-9]+(?:['’][a-z]+)?|[.!?]",
            normalized
        )

        # Keep only sentences containing at least one word.
        tokens = [token for token in tokens
                  if re.search(r"[a-z0-9]", token)]

        if tokens:
            sentences.append(tokens)

    return sentences


def flatten(sentences):
    return [word for sentence in sentences for word in sentence]


def split_corpus(sentences, test_ratio=0.2):
    """Split sentences into training and testing sets."""
    sentences = list(sentences)
    random.Random(RANDOM_SEED).shuffle(sentences)

    if len(sentences) < 2:
        return sentences, sentences

    test_size = max(1, int(len(sentences) * test_ratio))
    test_size = min(test_size, len(sentences) - 1)

    return sentences[test_size:], sentences[:test_size]


# ============================================================
# 3. N-GRAM LANGUAGE MODEL
# ============================================================

class NGramModel:
    """Unigram, bigram or trigram statistical language model."""

    def __init__(self, n, training_sentences):
        self.n = n
        self.counts = Counter()
        self.context_counts = Counter()
        self.training_sentences = training_sentences

        self.vocabulary = {
            word for sentence in training_sentences
            for word in sentence
        }
        self.vocabulary.add(UNK)

        self._build_model()

    def _build_model(self):
        """Calculate N-Gram and context frequencies."""
        for sentence in self.training_sentences:
            words = [
                word if word in self.vocabulary else UNK
                for word in sentence
            ]

            if self.n == 1:
                padded = words
            else:
                padded = [START] * (self.n - 1) + words + [END]

            for i in range(len(padded) - self.n + 1):
                gram = tuple(padded[i:i + self.n])
                self.counts[gram] += 1

                if self.n > 1:
                    context = gram[:-1]
                    self.context_counts[context] += 1

        # For conditional probabilities, START is context-only.
        # Vocabulary size excludes special context/end tokens.
        self.vocab_size = max(1, len(self.vocabulary) + 1)

    def probability(self, words, smoothing=True):
        """
        Return probability of a word or N-Gram.

        For bigrams/trigrams, this is the conditional probability
        of the last word given its preceding context.
        """
        gram = tuple(words)

        if len(gram) != self.n:
            raise ValueError(f"Expected {self.n} words.")

        count = self.counts[gram]

        if self.n == 1:
            total = sum(self.counts.values())

            if smoothing:
                return (count + 1) / (
                    total + self.vocab_size
                )

            return count / total if total else 0.0

        context = gram[:-1]
        context_count = self.context_counts[context]

        if smoothing:
            return (count + 1) / (
                context_count + self.vocab_size
            )

        if context_count == 0:
            return 0.0

        return count / context_count

    def sentence_log_probability(self, sentence, smoothing=True):
        """Calculate log probability without numerical underflow."""
        words = [
            word if word in self.vocabulary else UNK
            for word in sentence
        ]

        if self.n == 1:
            grams = [(word,) for word in words]
        else:
            padded = [START] * (self.n - 1) + words + [END]
            grams = [
                tuple(padded[i:i + self.n])
                for i in range(len(padded) - self.n + 1)
            ]

        log_probability = 0.0

        for gram in grams:
            p = self.probability(gram, smoothing)

            if p <= 0:
                return float("-inf")

            log_probability += math.log(p)

        return log_probability

    def sentence_probability(self, sentence, smoothing=True):
        """Return sentence probability, avoiding underflow."""
        log_p = self.sentence_log_probability(
            sentence, smoothing
        )

        if log_p == float("-inf"):
            return 0.0

        if log_p < -745:
            return 0.0

        return math.exp(log_p)

    def perplexity(self, test_sentences):
        """
        Calculate perplexity using smoothed probabilities.

        Lower perplexity generally indicates better prediction
        of the evaluation text.
        """
        total_log_probability = 0.0
        token_count = 0

        for sentence in test_sentences:
            log_p = self.sentence_log_probability(
                sentence, smoothing=True
            )

            if log_p == float("-inf"):
                return float("inf")

            total_log_probability += log_p
            token_count += len(sentence) + (1 if self.n > 1 else 0)

        if token_count == 0:
            return float("inf")

        return math.exp(-total_log_probability / token_count)

    def predict_next(self, context):
        """Predict the most probable next word."""
        context = [
            word.lower() for word in context.split()
        ]

        if self.n == 1:
            candidates = {
                gram[0] for gram in self.counts
            }
            prefix = ()
        else:
            required = self.n - 1
            prefix = tuple(context[-required:])

            candidates = {
                gram[-1]
                for gram in self.counts
                if gram[:-1] == prefix
            }

        # Avoid generating sentence-boundary symbols.
        candidates.discard(START)
        candidates.discard(END)

        if not candidates:
            candidates = self.vocabulary.copy()

        if not candidates:
            return UNK, 0.0

        best_word = max(
            candidates,
            key=lambda word: self.probability(
                prefix + (word,), smoothing=True
            ) if self.n > 1 else self.probability(
                (word,), smoothing=True
            )
        )

        if self.n == 1:
            p = self.probability((best_word,), True)
        else:
            p = self.probability(prefix + (best_word,), True)

        return best_word, p

    def generate_sentence(self, max_words=15):
        """Generate text using conditional N-Gram probabilities."""
        if self.n == 1:
            context = []
        else:
            context = [START] * (self.n - 1)

        generated = []

        for _ in range(max_words):
            if self.n == 1:
                candidates = [
                    (gram[0], count)
                    for gram, count in self.counts.items()
                    if gram[0] != START and gram[0] != END
                ]
            else:
                prefix = tuple(context[-(self.n - 1):])
                candidates = [
                    (gram[-1], count)
                    for gram, count in self.counts.items()
                    if gram[:-1] == prefix
                ]

            candidates = [
                (word, count) for word, count in candidates
                if word not in (START, END)
            ]

            if not candidates:
                break

            # Count-weighted sampling gives more frequent words
            # higher chances of being selected.
            words, weights = zip(*candidates)
            word = random.choices(words, weights=weights, k=1)[0]

            generated.append(word)

            if self.n > 1:
                context.append(word)

        return " ".join(generated)


# ============================================================
# 4. DISPLAY FREQUENCY TABLES
# ============================================================

def show_frequency_table(model, limit=20):
    print(f"\nTop {limit} N-Grams — {model.n}-Gram Model")
    print("-" * 55)
    print(f"{'N-Gram':35} {'Frequency':>12}")

    for gram, count in model.counts.most_common(limit):
        print(f"{' '.join(gram):35} {count:>12}")


# ============================================================
# 5. SENTENCE PROBABILITY AND SMOOTHING
# ============================================================

def compare_sentence_probabilities(models, sentence):
    print("\nSENTENCE PROBABILITY COMPARISON")
    print("=" * 75)
    print("Sentence:", " ".join(sentence))

    for n, model in models.items():
        raw = model.sentence_probability(
            sentence, smoothing=False
        )
        smooth = model.sentence_probability(
            sentence, smoothing=True
        )

        print(f"\n{n}-Gram:")
        print(f"  Without smoothing: {raw:.8e}")
        print(f"  With Laplace smoothing: {smooth:.8e}")

        if raw == 0 and smooth > 0:
            print("  Smoothing assigned a non-zero probability.")


# ============================================================
# 6. NEXT-WORD PREDICTION ACCURACY
# ============================================================

def prediction_accuracy(model, test_sentences):
    """
    Evaluate top-1 next-word prediction on test sequences.

    Context length is n-1. Unigram prediction uses no context.
    """
    correct = 0
    total = 0

    for sentence in test_sentences:
        for i in range(len(sentence) - 1):
            if model.n == 1:
                context = ""
            else:
                context = " ".join(
                    sentence[max(0, i - model.n + 2):i + 1]
                )

            actual = sentence[i + 1]
            predicted, _ = model.predict_next(context)

            if predicted == actual:
                correct += 1

            total += 1

    return (100.0 * correct / total) if total else 0.0


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

def compare_models(models, test_sentences):
    print("\nEXPERIMENTAL ANALYSIS")
    print("=" * 85)
    print(
        f"{'Model':<12}{'Perplexity':>18}"
        f"{'Prediction Accuracy':>25}"
    )
    print("-" * 85)

    for n, model in models.items():
        perplexity = model.perplexity(test_sentences)
        accuracy = prediction_accuracy(model, test_sentences)

        pp = (
            f"{perplexity:.4f}"
            if math.isfinite(perplexity) else "Infinity"
        )

        print(
            f"{n}-Gram{'':<6}{pp:>18}"
            f"{accuracy:>23.2f}%"
        )

    print("\nInterpretation:")
    print("- Lower perplexity usually indicates better prediction.")
    print("- Higher next-word accuracy indicates more correct predictions.")
    print("- Higher N captures more context but may increase sparsity.")
    print("- Results depend on the corpus and its size.")


# ============================================================
# 8. INTERACTIVE EXPERIMENT MENU
# ============================================================

def main():
    print("=" * 75)
    print("NLP EXPERIMENT 1: N-GRAM LANGUAGE MODELS")
    print("=" * 75)
    print("Dataset folder:", DATA_DIR)

    try:
        files = discover_files()
    except FileNotFoundError as error:
        print(error)
        return

    if not files:
        print("No supported text files found in mobiletext.")
        return

    print(f"\nFound {len(files)} possible dataset files.")

    print("\nChoose your training dataset.")
    train_path = choose_file(files, "AVAILABLE DATASETS")

    if train_path is None:
        print("No training dataset selected.")
        return

    print("\nChoose a separate test dataset, or use a random split.")
    print("Enter 0 to use an automatic 80/20 split.")
    test_path = choose_file(files, "TEST DATASET")

    print("\nLoading training corpus...")
    train_text = read_text_file(train_path, MAX_TRAIN_CHARS)
    raw_train_sentences = tokenize_sentences(train_text)

    if not raw_train_sentences:
        print("No usable sentences found in the training file.")
        return

    if test_path is not None:
        print("Loading test corpus...")
        test_text = read_text_file(test_path, MAX_TEST_CHARS)
        test_sentences = tokenize_sentences(test_text)

        if not test_sentences:
            print("Test file contained no usable sentences.")
            return

        train_sentences = raw_train_sentences
        print("Training file:", train_path.name)
        print("Testing file:", test_path.name)
    else:
        train_sentences, test_sentences = split_corpus(
            raw_train_sentences
        )
        print("Using an 80/20 random sentence split.")

    print("\nPREPROCESSING SUMMARY")
    print("-" * 55)
    print("Training sentences:", len(train_sentences))
    print("Testing sentences:", len(test_sentences))
    print("Training tokens:", len(flatten(train_sentences)))
    print("Testing tokens:", len(flatten(test_sentences)))
    print("Sample normalized sentence:")
    print(" ".join(train_sentences[0]))

    print("\nBuilding unigram, bigram and trigram models...")
    models = {
        n: NGramModel(n, train_sentences)
        for n in (1, 2, 3)
    }

    print("\nAll models trained successfully.")

    while True:
        print("\n" + "=" * 65)
        print("EXPERIMENT MENU")
        print("=" * 65)
        print("1. Show corpus and preprocessing summary")
        print("2. Display unigram, bigram and trigram frequency tables")
        print("3. Calculate sentence probability before/after smoothing")
        print("4. Evaluate perplexity and prediction accuracy")
        print("5. Predict the next word")
        print("6. Generate sentences")
        print("7. Compare all models")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ").strip()

        if choice == "1":
            print("\nTraining dataset:", train_path.name)
            print("Training sentences:", len(train_sentences))
            print("Training vocabulary:", len(models[1].vocabulary))
            print("\nSample sentences:")
            for sentence in train_sentences[:5]:
                print(" ", " ".join(sentence))

        elif choice == "2":
            for model in models.values():
                show_frequency_table(model)

        elif choice == "3":
            text = input("Enter a sentence: ").strip()
            sentence = tokenize_sentences(text)

            if sentence:
                compare_sentence_probabilities(
                    models, sentence[0]
                )
            else:
                print("Please enter a valid sentence.")

        elif choice == "4":
            for n, model in models.items():
                pp = model.perplexity(test_sentences)
                print(
                    f"{n}-Gram perplexity: "
                    f"{pp:.4f}" if math.isfinite(pp)
                    else f"{n}-Gram perplexity: Infinity"
                )

        elif choice == "5":
            text = input("Enter a word sequence: ").strip()

            for n, model in models.items():
                word, probability = model.predict_next(text)
                print(
                    f"{n}-Gram prediction: {word} "
                    f"(conditional probability: {probability:.6f})"
                )

        elif choice == "6":
            try:
                count = int(input(
                    "Number of words to generate (default 15): "
                ) or "15")
                count = max(1, min(count, 50))
            except ValueError:
                count = 15

            for n, model in models.items():
                print(
                    f"{n}-Gram generated text:\n"
                    f"{model.generate_sentence(count)}\n"
                )

        elif choice == "7":
            compare_models(models, test_sentences)

        elif choice == "8":
            print("Experiment completed. Goodbye!")
            break

        else:
            print("Invalid choice. Select a number from 1 to 8.")


if __name__ == "__main__":
    main()
