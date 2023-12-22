# the book we're going to split and embed
# INPUT_FILES = ["./ambrosetti.pdf"]
INPUT_FILES = ["database-concepts.pdf", "oracle-database-23c-new-features-guide.pdf"]
# INPUT_FILES = ["./python4everybody.pdf"]

# Cohere embeddings model
# for english use this one
EMBED_MODEL = "cohere.embed-english-v3.0"
# used for token counting
TOKENIZER = "Cohere/command-nightly"
# for other language (must be consistent when do query)
# EMBED_MODEL = "cohere.embed-multilingual-v3.0"

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64
