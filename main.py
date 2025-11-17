import argparse
from rag.pipeline import ingest, rag_query

def main():
    parser = argparse.ArgumentParser(description="Multi-file RAG Pipeline")

    parser.add_argument(
        "--ingest",
        type=str,
        help="Path to file or URL to ingest into vector database"
    )

    parser.add_argument(
        "--ask",
        type=str,
        help="Question to query the RAG system"
    )

    args = parser.parse_args()

    if args.ingest:
        print(f"[INFO] Ingesting: {args.ingest}")
        chunks = ingest(args.ingest)
        print(f"[SUCCESS] Ingested {chunks} chunks.\n")

    if args.ask:
        print(f"[QUESTION] {args.ask}")
        answer = rag_query(args.ask)
        print(f"\n[ANSWER]\n{answer}\n")


if __name__ == "__main__":
    main()
