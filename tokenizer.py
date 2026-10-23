
import re
import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def train(self, text, vocab_size=300):
        # Start with characters and a word-ending marker.
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        word_counts = Counter(words)

        sequences = {
            word: list(word) + ["</w>"]
            for word in word_counts
        }

        self.merges = []

        # Initial vocabulary consists of individual symbols.
        symbols = set()
        for sequence in sequences.values():
            symbols.update(sequence)

        # Repeatedly merge the most frequent adjacent pair.
        while len(symbols) < vocab_size:
            pair_counts = Counter()

            for word, sequence in sequences.items():
                for i in range(len(sequence) - 1):
                    pair_counts[(sequence[i], sequence[i + 1])] += word_counts[word]

            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < 2:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            for word, sequence in sequences.items():
                new_sequence = []
                i = 0

                while i < len(sequence):
                    if (
                        i < len(sequence) - 1
                        and (sequence[i], sequence[i + 1]) == best_pair
                    ):
                        new_sequence.append(merged_symbol)
                        i += 2
                    else:
                        new_sequence.append(sequence[i])
                        i += 1

                sequences[word] = new_sequence

            symbols.add(merged_symbol)

        # Reserve an unknown-token ID.
        self.vocab = {
            symbol: i for i, symbol in enumerate(sorted(symbols))
        }

        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = len(self.vocab)

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = []

        for word in words:
            sequence = list(word) + ["</w>"]

            for left, right in self.merges:
                merged = left + right
                new_sequence = []
                i = 0

                while i < len(sequence):
                    if (
                        i < len(sequence) - 1
                        and sequence[i] == left
                        and sequence[i + 1] == right
                    ):
                        new_sequence.append(merged)
                        i += 2
                    else:
                        new_sequence.append(sequence[i])
                        i += 1

                sequence = new_sequence

            for symbol in sequence:
                token_ids.append(self.vocab.get(symbol, self.vocab["<unk>"]))

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        id_to_token = {i: token for token, i in self.vocab.items()}
        pieces = [id_to_token.get(i, "<unk>") for i in token_ids]

        if skip_special_tokens:
            pieces = [piece for piece in pieces if piece not in {"<pad>", "<eos>"}]

        text = "".join(
            piece.replace("</w>", " ")
            for piece in pieces
        )

        return text.strip()

    def save(self, path="bpe_tokenizer.json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "vocab": self.vocab,
                    "merges": self.merges
                },
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path="bpe_tokenizer.json"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = data["vocab"]
        self.merges = [tuple(pair) for pair in data["merges"]]


if __name__ == "__main__":
    with open("training_data.txt", "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=300)
    tokenizer.save()

    sample = "transformers help developers learn"
    encoded = tokenizer.encode(sample)

    print("Vocabulary size:", len(tokenizer.vocab))
    print("Original text:", sample)
    print("Token IDs:", encoded)
    print("Decoded text:", tokenizer.decode(encoded))
