"""Data preparation for VoidGPT 1 120K.

Supports two data sources:
1. Built-in synthetic knowledge entries (non-story educational text)
2. Real datasets from HuggingFace: nano_wiki, wikitext-2, simple_wikipedia

No fiction, no stories — only factual encyclopedia-style text.

Output: data/train.txt
"""

import argparse
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Non-story educational content in simple English.
# Each entry is a short factual paragraph or Q&A pair.
# Designed for tiny model training: simple vocabulary, short sentences.

KNOWLEDGE_ENTRIES = [
    # Science
    "Water is a substance made of two hydrogen atoms and one oxygen atom. Its chemical formula is H2O. Water covers about seventy percent of the Earth surface. Water is essential for all known forms of life. Water can exist as solid ice, liquid water, or gas called water vapor.",
    "The Sun is a star at the center of our solar system. The Sun is a giant ball of hot gas. It produces light and heat through nuclear fusion. The Sun is about 150 million kilometers from Earth. The Sun is the source of almost all energy on Earth.",
    "Photosynthesis is the process by which plants make food. Plants use sunlight, water, and carbon dioxide to create sugar. This process happens in the leaves of plants. The green color in leaves comes from chlorophyll. Plants release oxygen as a byproduct of photosynthesis.",
    "Gravity is a force that pulls objects toward each other. The strength of gravity depends on mass. Larger objects have stronger gravity. Gravity keeps the planets in orbit around the Sun. Gravity is what makes things fall to the ground on Earth.",
    "An atom is the smallest unit of matter. Atoms are made of protons, neutrons, and electrons. Protons have positive charge. Electrons have negative charge. Neutrons have no charge. The center of an atom is called the nucleus.",
    "DNA is a molecule that carries genetic information. DNA is found in the cells of all living things. DNA is shaped like a double helix, like a twisted ladder. DNA contains the instructions for how an organism looks and functions. DNA is passed from parents to children.",
    "The speed of light is about 300 million meters per second. Light is the fastest thing in the universe. Nothing can travel faster than light. Light from the Sun takes about 8 minutes to reach Earth. The speed of light is a fundamental constant in physics.",
    "A magnet is an object that creates a magnetic field. Magnets attract certain metals like iron. Every magnet has two ends called poles: north and south. Opposite poles attract each other. Same poles push each other away. The Earth itself is a giant magnet.",
    "Electricity is the flow of electric charge. Electricity flows through conductors like copper wire. Materials that block electricity are called insulators. Voltage is the pressure that pushes electricity. Current is the amount of electricity flowing. Power equals voltage times current.",
    "A cell is the basic unit of life. All living things are made of cells. Some organisms have only one cell. Humans have trillions of cells. Cells have a membrane that protects them. Inside the cell is the nucleus which contains DNA.",

    # History
    "The ancient Egyptians built pyramids as tombs for their kings. The largest pyramid is the Great Pyramid of Giza. It was built about 4500 years ago. The Egyptians also developed writing called hieroglyphics. The Nile River was central to Egyptian civilization.",
    "The Roman Empire was one of the largest empires in history. Rome was founded as a small city and grew into a huge empire. The Romans built roads, bridges, and aqueducts. The Roman Empire fell in the year 476. Latin was the language of the Romans.",
    "World War II was a global war that lasted from 1939 to 1945. It involved most of the world nations. The war was fought between the Allies and the Axis powers. The Allies included the United States, Britain, and the Soviet Union. The war ended when Japan surrendered in 1945.",
    "The printing press was invented by Johannes Gutenberg around 1440. Before the printing press, books were copied by hand. The printing press made books cheaper and more available. This led to more people learning to read. The printing press changed the world.",
    "The Industrial Revolution started in Britain in the 1700s. It changed how goods were made. Machines replaced hand labor. Factories were built. Steam engines provided power. Railroads were built to transport goods. The Industrial Revolution led to modern industry.",

    # Geography
    "The Earth has seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America. Asia is the largest continent. Africa is the second largest. Antarctica is the coldest continent. The Pacific Ocean is the largest ocean on Earth.",
    "A mountain is a large natural elevation of the Earth surface. Mount Everest is the tallest mountain on Earth. It is 8848 meters high. Mountains form when tectonic plates push against each other. The process is called mountain building or orogeny.",
    "A river is a natural flowing watercourse. Rivers usually flow toward an ocean or sea. The Nile is the longest river in the world. The Amazon is the second longest but has the most water. Rivers provide water for drinking, farming, and transportation.",
    "A desert is a dry area with very little rain. The Sahara is the largest hot desert in the world. Deserts can be hot or cold. Antarctica is actually the largest desert in the world. Desert plants and animals are adapted to survive with little water.",

    # Mathematics
    "Addition is combining two or more numbers to get a sum. The plus sign is used for addition. For example, two plus three equals five. Addition is one of the four basic operations in math. The others are subtraction, multiplication, and division.",
    "Multiplication is repeated addition. The times sign is used for multiplication. For example, three times four means adding three four times. Three times four equals twelve. Multiplication is one of the basic operations in mathematics.",
    "A triangle is a shape with three sides and three angles. The sum of angles in a triangle is always 180 degrees. A triangle with all sides equal is called equilateral. A triangle with two equal sides is called isosceles. A triangle with no equal sides is called scalene.",
    "A circle is a round shape. Every point on a circle is the same distance from the center. That distance is called the radius. The distance across a circle is called the diameter. The diameter is twice the radius. The area of a circle is pi times radius squared.",

    # Biology
    "The heart is a muscle that pumps blood through the body. The heart has four chambers. Blood carries oxygen and nutrients to cells. The heart beats about 100 thousand times every day. Blood vessels called arteries carry blood away from the heart.",
    "The brain is the control center of the body. The brain is inside the skull. The brain has three main parts: the cerebrum, the cerebellum, and the brainstem. The cerebrum handles thinking and memory. The cerebellum controls movement. The brainstem controls breathing.",
    "A food chain shows how energy moves through nature. Plants make food from sunlight. Animals that eat plants are called herbivores. Animals that eat other animals are called carnivores. Animals that eat both are called omnivores. Humans are omnivores.",
    "Evolution is the process by which living things change over time. Charles Darwin developed the theory of evolution. Evolution happens through natural selection. Organisms that are better adapted survive and reproduce. Evolution takes place over many generations.",

    # Technology
    "A computer is a machine that processes information. Computers use binary code made of zeros and ones. The main parts of a computer are the processor, memory, and storage. The processor does calculations. Memory holds data temporarily. Storage keeps data permanently.",
    "The internet is a global network of computers. Computers connect to each other to share information. The World Wide Web is part of the internet. Websites are pages on the web. The internet was developed in the 1960s and 1970s. Today billions of people use the internet.",
    "A programming language is a way to give instructions to a computer. Python is a popular programming language. Code is written as text that the computer can understand. Programs are collections of code that perform tasks. Programming is how software is created.",

    # Simple Q&A format
    "Question: What is the chemical formula for water?\nAnswer: The chemical formula for water is H2O. This means each water molecule has two hydrogen atoms and one oxygen atom.",
    "Question: What is the largest planet in our solar system?\nAnswer: Jupiter is the largest planet in our solar system. It is a gas giant and is more than twice as massive as all other planets combined.",
    "Question: What is photosynthesis?\nAnswer: Photosynthesis is the process where plants use sunlight to make food from water and carbon dioxide. Plants release oxygen during this process.",
    "Question: What is the speed of light?\nAnswer: The speed of light is about 300 million meters per second. It is the fastest speed possible in the universe.",
    "Question: What are the seven continents?\nAnswer: The seven continents are Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
    "Question: What is DNA?\nAnswer: DNA is a molecule that carries genetic information in living things. It is shaped like a double helix and contains instructions for how organisms grow and function.",
    "Question: What is the largest ocean?\nAnswer: The Pacific Ocean is the largest ocean on Earth. It covers about one third of the Earth surface.",
    "Question: What is gravity?\nAnswer: Gravity is a force that pulls objects toward each other. It depends on mass and distance. Gravity keeps planets in orbit and makes things fall.",
    "Question: What is the brain?\nAnswer: The brain is the control center of the body. It handles thinking, memory, movement, and basic functions like breathing.",
    "Question: What is a computer?\nAnswer: A computer is a machine that processes information using binary code. Its main parts are the processor, memory, and storage.",
]


def generate_dataset(num_repeats: int = 50) -> str:
    """Generate training text by repeating and shuffling knowledge entries.

    Args:
        num_repeats: how many times to repeat the entries

    Returns:
        Generated text string
    """
    rng = random.Random(42)
    all_entries = []

    for _ in range(num_repeats):
        entries = list(KNOWLEDGE_ENTRIES)
        rng.shuffle(entries)
        all_entries.extend(entries)

    return "\n\n".join(all_entries) + "\n"


def load_nano_wiki(max_chars: int = 500_000) -> str:
    """Load nano_wiki dataset from HuggingFace.

    Synthetic encyclopedia-style text in simple English.
    ~2.9M tokens, designed for tiny LMs.
    """
    from datasets import load_dataset

    ds = load_dataset("sixf0ur/nano_wiki", split="train")
    texts = []
    total = 0
    for item in ds:
        title = item.get("title", "")
        text = item.get("text", "")
        entry = f"{title}. {text}" if title else text
        texts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    return "\n\n".join(texts)


def load_wikitext2(max_chars: int = 500_000) -> str:
    """Load WikiText-2 dataset from HuggingFace.

    Real Wikipedia articles, verified Good and Featured quality.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = []
    total = 0
    for item in ds:
        text = item["text"].strip()
        if len(text) > 50:  # skip empty/short lines
            texts.append(text)
            total += len(text)
            if total >= max_chars:
                break
    return "\n\n".join(texts)


def load_simple_wikipedia_qa(max_chars: int = 500_000) -> str:
    """Load simple Wikipedia Q&A dataset from HuggingFace.

    Q&A pairs based on Simple English Wikipedia articles.
    """
    from datasets import load_dataset

    ds = load_dataset("Lovre/simple-wikipedia-qa-433k", split="train")
    texts = []
    total = 0
    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if question and answer:
            entry = f"Question: {question}\nAnswer: {answer}"
            texts.append(entry)
            total += len(entry)
            if total >= max_chars:
                break
    return "\n\n".join(texts)


REAL_DATASETS = {
    "nano_wiki": load_nano_wiki,
    "wikitext2": load_wikitext2,
    "simple_wiki_qa": load_simple_wikipedia_qa,
}


def load_real_mixed(max_chars: int = 500_000) -> str:
    """Load a mix of real datasets (nano_wiki + wikitext2) for diverse training.

    No synthetic data — pure real text from multiple sources to maximize
    diversity and reduce overfitting from repetition.
    """
    half = max_chars // 2
    texts = []
    try:
        wiki = load_nano_wiki(max_chars=half)
        texts.append(wiki)
        print(f"  nano_wiki: {len(wiki)} chars")
    except Exception as e:
        print(f"  nano_wiki failed: {e}")
    try:
        wt2 = load_wikitext2(max_chars=half)
        texts.append(wt2)
        print(f"  wikitext2: {len(wt2)} chars")
    except Exception as e:
        print(f"  wikitext2 failed: {e}")
    return "\n\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for VoidGPT 1 120K")
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "nano_wiki", "wikitext2", "simple_wiki_qa", "mixed", "real_mixed"],
        help="Data source: synthetic, real dataset, mixed (synthetic+nano_wiki), or real_mixed (nano_wiki+wikitext2)",
    )
    parser.add_argument("--max_chars", type=int, default=500_000, help="Max chars for real datasets")
    parser.add_argument("--num_repeats", type=int, default=50, help="Repeats for synthetic data")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DATA_DIR / "train.txt"

    if args.source == "synthetic":
        text = generate_dataset(num_repeats=args.num_repeats)
        print(f"Source: synthetic ({len(KNOWLEDGE_ENTRIES)} entries × {args.num_repeats} repeats)")
    elif args.source == "mixed":
        synthetic = generate_dataset(num_repeats=args.num_repeats)
        print(f"Synthetic: {len(synthetic)} chars")
        try:
            real = load_nano_wiki(max_chars=args.max_chars)
            print(f"nano_wiki: {len(real)} chars")
            text = synthetic + "\n\n" + real
        except Exception as e:
            print(f"Could not load nano_wiki ({e}), using synthetic only")
            text = synthetic
        print(f"Source: mixed (synthetic + nano_wiki)")
    elif args.source == "real_mixed":
        text = load_real_mixed(max_chars=args.max_chars)
        print(f"Source: real_mixed (nano_wiki + wikitext2)")
    else:
        loader = REAL_DATASETS[args.source]
        text = loader(max_chars=args.max_chars)
        print(f"Source: {args.source} ({len(text)} chars)")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Dataset written to {output_path}")
    print(f"Size: {len(text)} characters")
    print(f"Unique characters: {len(set(text))}")


if __name__ == "__main__":
    main()
