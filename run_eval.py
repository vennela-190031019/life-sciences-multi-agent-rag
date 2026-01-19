import json
from rag_basic import build_retriever
from graph_agents import make_graph

def main():
    retriever = build_retriever(k=3)
    graph = make_graph(retriever)
    print("GRAPH VALUE:", graph)


    with open("eval_set.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    passed = 0
    for t in tests:
        q = t["q"]
        expected = t.get("expected_agent")

        result = graph.invoke({
            "question": q,
            "agent": "general",
            "context": "",
            "citations": [],
            "answer": ""
        })

        got = result.get("agent")
        ok = (got == expected) if expected else True

        print("\nQ:", q)
        print("Agent:", got, "| Expected:", expected, "| PASS" if ok else "| FAIL")
        print("Sources:", sorted({c.get("source") for c in (result.get("citations") or [])}))
        print("Answer:", (result.get("answer") or "")[:180], "..." if len(result.get("answer") or "") > 180 else "")

        if ok:
            passed += 1

    print(f"\n✅ Passed {passed}/{len(tests)} routing checks")

if __name__ == "__main__":
    main()
