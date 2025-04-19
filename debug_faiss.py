from retriever import load_faiss_index
import sys

if len(sys.argv) < 3:
    print("Usage: python debug_faiss.py <course_id> <section>")
    sys.exit(1)

course_id = sys.argv[1]
section = sys.argv[2]

try:
    index, metadata = load_faiss_index(section, course_id)
except FileNotFoundError as e:
    print(f"\n[ERROR] {e}")
    sys.exit(1)

print(f"\n🔍 {len(metadata)} chunks found in section: {section} (Course ID: {course_id})")

for i, m in enumerate(metadata):
    if isinstance(m, dict):
        chunk = m.get("text", "")
    else:
        chunk = m  # fallback if plain string

    print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}")
