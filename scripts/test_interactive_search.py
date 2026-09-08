from backend.retrieval import retriever

print("=" * 70)
print("TERRAWATCH AI - INTERACTIVE SEMANTIC SEARCH")
print("=" * 70)

while True:
    query = input("\nEnter your search query (or 'exit' to quit): ")

    if query.lower() == "exit":
        break

    results = retriever.search_by_text(query, top_k=5)

    print(f"\nSearch results for: '{query}'")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        print(f"{i}. Image ID : {result['image_id']}")
        print(f"   Score    : {result['score']:.4f}")
        print(f"   Image    : {result['path']}")
        print()
