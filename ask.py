# ask.py

import argparse
import traceback
from qa_engine_rag import answer_question_rag

def main():
    parser = argparse.ArgumentParser(description="Ask a natural language question about your Canvas course.")
    parser.add_argument("question", type=str, nargs='+', help="Your question (e.g. 'When are office hours for IS 327')")
    args = parser.parse_args()

    # Join question parts into a full string
    question = " ".join(args.question).strip()

    print("\nThinking...\n")

    try:
        answer = answer_question_rag(question)
        if not answer or not answer.strip():
            print("[ERROR] No answer generated.")
        else:
            print("\nAnswer:\n" + answer)

    except Exception as e:
        print("\n[ERROR] Something went wrong:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
