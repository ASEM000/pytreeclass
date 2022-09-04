from __future__ import annotations

import requests

# def save_viz(tree, filename, method="tree_mermaid_md"):

#     if method == "tree_mermaid_md":
#         FMT = "```mermaid\n" + tree_mermaid(tree) + "\n```"

#         with open(f"{filename}.md", "w") as f:
#             f.write(FMT)

#     elif method == "tree_mermaid_html":
#         FMT = "<html><body><script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script>"
#         FMT += "<script>mermaid.initialize({ startOnLoad: true });</script><div class='mermaid'>"
#         FMT += tree_mermaid(tree)
#         FMT += "</div></body></html>"

#         with open(f"{filename}.html", "w") as f:
#             f.write(FMT)

#     elif method == "tree_diagram":
#         with open(f"{filename}.txt", "w") as f:
#             f.write(tree_diagram(tree))

#     elif method == "tree_box":
#         with open(f"{filename}.txt", "w") as f:
#             f.write(tree_box(tree))

#     elif method == "summary":
#         with open(f"{filename}.txt", "w") as f:
#             f.write(tree_summary(tree))


def _generate_mermaid_link(mermaid_string: str) -> str:
    """generate a one-time link mermaid diagram"""
    url_val = "https://pytreeclass.herokuapp.com/generateTemp"
    request = requests.post(url_val, json={"description": mermaid_string})
    generated_id = request.json()["id"]
    generated_html = f"https://pytreeclass.herokuapp.com/temp/?id={generated_id}"
    return f"Open URL in browser: {generated_html}"


# def _mermaid_table_row(node):
#     return "<tr>" "<td>" f"{node}" "</td>" "<tr>"


# def _mermaid_table(node):
#     return "<div align=" "left" ">" "<table>" f"{node}" "</table>" "</div>"
