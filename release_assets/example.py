"""Run with: python example.py /path/to/downloaded/snapshot"""
import sys
from reranker import Reranker


def main(path):
    reranker = Reranker(path, device='cpu')
    result = reranker.rank('What is a reranker?', [
        'A reranker assigns relevance scores to query-document pairs.',
        'Tomatoes grow best in sunny gardens.',
        'A ranking model sorts candidate documents by a raw score.',
    ], batch_size=2)
    print(result)
    return result


if __name__ == '__main__':
    main(sys.argv[1])
