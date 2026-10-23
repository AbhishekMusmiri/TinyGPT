
import re
import json
from collections import Counter


class ImprovedBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def train(self, text, vocab_size=300):
        # Split text into words and punctuation.
        words = re.findall(r"\w+|[^\w\s]", text.lower())

        # Represent each word as characters, marking its ending.
        word_counts = Counter(tuple(list(word) + ["</w>"]) for word in words)

        symbols = set()
        for word in word_counts:
            symbols.update(word)

        self.merges = []

        while len(symbols) < vocab_size:
            pair_counts = Counter()

            for word, frequency in word_counts.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += frequency

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]

            if count < 2:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            self.merges.append(list(best_pair))

            new_word_counts = Counter()

            for word, frequency in word_counts.items():
                new_word = []
                i = 0

                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(merged_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_word_counts[tuple(new_word)] += frequency

            word_counts = new_word_counts
            symbols.add(merged_symbol)

        self.vocab = {symbol: i for i, symbol in enumerate(sorted(symbols))}
        self.vocab["<unk>"] = len(self.vocab)
        return self

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = []

        for word in words:
            pieces = list(word) + ["</w>"]

            for left, right in self.merges:
                merged = []
                i = 0

                while i < len(pieces):
                    if (
                        i < len(pieces) - 1
                        and pieces[i] == left
                        and pieces[i + 1] == right
                    ):
                        merged.append(left + right)
                        i += 2
                    else:
                        merged.append(pieces[i])
                        i += 1

                pieces = merged

            unk_id = self.vocab["<unk>"]
            token_ids.extend(self.vocab.get(piece, unk_id) for piece in pieces)

        return token_ids

    def decode(self, token_ids):
        reverse_vocab = {i: token for token, i in self.vocab.items()}
        pieces = [reverse_vocab.get(i, "<unk>") for i in token_ids]
        text = "".join(pieces)
        return text.replace("</w>", " ").strip()

    def save(self, path="improved_bpe_tokenizer.json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {"vocab": self.vocab, "merges": self.merges},
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path="improved_bpe_tokenizer.json"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = data["vocab"]
        self.merges = data["merges"]
        return self


if __name__ == "__main__":
    with open("training_data.txt", "r", encoding="utf-8") as file:
        training_text = file.read()

    tokenizer = ImprovedBPETokenizer()
    tokenizer.train(training_text, vocab_size=300)
    tokenizer.save()

    sample = "machine learning finds patterns in data"
    ids = tokenizer.encode(sample)

    print("Vocabulary size:", len(tokenizer.vocab))
    print("Original:", sample)
    print("Token IDs:", ids)
    print("Token count:", len(ids))
    print("Decoded:", tokenizer.decode(ids))
