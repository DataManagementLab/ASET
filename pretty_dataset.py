import logging.config

import tqdm

import datasets.countries.countries as dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

COLORS = [
    "#f6b93b",
    "#f8c291",
    "#6a89cc",
    "#82ccdd",
    "#b8e994",
    "#f6b93b",
    "#e55039",
    "#4a69bd",
    "#60a3bc",
    "#78e08f",
    "#fa983a",
    "#eb2f06",
    "#1e3799",
    "#3c6382",
    "#38ada9",
    "#e58e26",
    "#b71540",
    "#0c2461",
    "#0a3d62",
    "#079992"
]

if __name__ == "__main__":
    documents = dataset.load_dataset()

    for document in tqdm.tqdm(documents):
        lines = [
            "<!DOCTYPE html><html lang=\"de-de\">"
            f"<head><title>{document['id']}</title>"
            "<meta charset=\"UTF-8\">"
            "<style>* {font-family: sans-serif;}</style>"
            "</head><body>\n"
            f"<h1>{document['id']}</h1>"
        ]

        # create a sorted list of all highlights
        mappings = {}
        highlights = []
        for attribute, color in zip(dataset.ATTRIBUTES, COLORS):
            mappings[attribute] = color
            for mention in document["mentions"][attribute]:
                highlights.append({
                    "color": color,
                    "italic": False,
                    "start": mention["start_char"],
                    "end": mention["end_char"],
                    "opening-length": 43,
                    "closing-length": 11
                })
        for attribute, color in zip(dataset.ATTRIBUTES, COLORS):
            for mention in document["mentions_same_attribute_class"][attribute]:
                highlights.append({
                    "color": color,
                    "italic": True,
                    "start": mention["start"],
                    "end": mention["start"] + mention["length"],
                    "opening-length": 46,
                    "closing-length": 15
                })

        highlights = sorted(highlights, key=lambda x: x["start"])

        # highlight in the text
        remaining_highlights = highlights.copy()
        open_highlights = []
        characters = list(document["text"])
        new_characters = []
        offset = 0
        for idx in range(len(characters) + 1):
            while len(open_highlights) > 0 and open_highlights[0]["end"] <= idx:
                highlight = open_highlights[0]
                marker = f"</b>{'</i>' if highlight['italic'] else ''}</span>"
                new_characters += list(marker)
                open_highlights = open_highlights[1:]

            while len(remaining_highlights) > 0 and remaining_highlights[0]["start"] <= idx:
                highlight = remaining_highlights[0]
                marker = f"<span style='background-color: {highlight['color']}'>{'<i>' if highlight['italic'] else ''}<b>"
                new_characters += list(marker)
                open_highlights.append(highlight)
                open_highlights = sorted(open_highlights, key=lambda x: x["end"])
                remaining_highlights = remaining_highlights[1:]

            if idx < len(characters):
                new_characters.append(characters[idx])

        lines.append("".join(new_characters))

        lines.append("<ul>")
        for attribute, color in zip(dataset.ATTRIBUTES, COLORS):
            lines.append(f"<li><span style='color: {color}'><b>{attribute}</b></span></li>")
        lines.append("</ul>")

        lines.append("<b>mention</b>")
        lines.append("<b><i>mention_diff_value</i></b>")

        lines.append("</body>")

        with open(f"datasets/countries/html-documents/{document['id']}.html", "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
