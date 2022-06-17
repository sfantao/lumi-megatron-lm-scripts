import json
import argparse
import os
from typing import Iterable, Dict


def read_docs(fn: str) -> Iterable[str]:
    with open(fn) as fh:
        for line in fh:
            yield line


def doc_to_dict(doc: str, source: str) -> Dict[str, str]:
    return {"src": source, "text": doc}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.outfile, "w") as fout:
        for doc in read_docs(args.infile):
            print(json.dumps(doc_to_dict(doc, os.path.basename(args.infile))), file=fout)


if __name__ == "__main__":
    main()
