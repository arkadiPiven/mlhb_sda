#!/usr/bin/env python
import json
import networkx as nx
from collections import defaultdict

def convert_pairwise_to_rankings(pairwise_data):
    """
    Converts pairwise preference data into a single ranking for each prompt.

    pairwise_data: list of dict, each has:
      - "prompt" (str)
      - "human_preference" (list[int], length=2), e.g. [0,1] means second image is preferred
      - "image_path" (list[str], length=2), the two images being compared

    Returns: list of dict, each with:
      - "prompt" (str)
      - "image_path" (list[str]): unique images for that prompt in an order
      - "rank" (list[int]): ranks for each image (0=best, 1=2nd best, etc.)
    """
    grouped = defaultdict(list)
    for entry in pairwise_data:
        prompt = entry["prompt"]
        grouped[prompt].append(entry)

    ranked_data = []
    for prompt, entries in grouped.items():
        # Collect all distinct images
        image_set = set()
        for e in entries:
            for img in e["image_path"]:
                image_set.add(img)

        # Build directed graph: A->B if B is preferred to A
        G = nx.DiGraph()
        G.add_nodes_from(image_set)

        for e in entries:
            imgA, imgB = e["image_path"]
            prefA, prefB = e["human_preference"]
            # [0,1] => image_path[1] is preferred => edge from A->B
            if prefA == 1 and prefB == 0:
                G.add_edge(imgB, imgA)
            elif prefA == 0 and prefB == 1:
                G.add_edge(imgA, imgB)
            else:
                # tie or unexpected format => skip or handle differently
                pass

        try:
            # Topological sort => from least preferred to most preferred
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            print(f"Warning: cycle/contradiction in prompt: {prompt}")
            topo_order = list(image_set)  # fallback

        # Reverse => best first
        best_first = topo_order[::-1]

        # Build a dictionary: {img: rank_position}
        rank_dict = {img: i for i, img in enumerate(best_first)}

        # Sort images by ascending rank => best -> worst
        final_image_list = sorted(list(image_set), key=lambda x: rank_dict[x])
        final_rank_list = [rank_dict[img] for img in final_image_list]

        ranked_data.append({
            "prompt": prompt,
            "image_path": final_image_list,
            "rank": final_rank_list
        })

    return ranked_data


def main():
    # 1) Load your pairwise data from JSON, e.g. 'train.json'
    pairwise_json_path = "../HPDv2/train/train.json"  # adjust path as needed
    with open(pairwise_json_path, "r") as f:
        pairwise_data = json.load(f)

    # 2) Convert to ranking format
    ranked_data = convert_pairwise_to_rankings(pairwise_data)

    # 3) Save to 'ranked_train.json'
    ranked_json_path = "../HPDv2/train/ranked_train.json"
    with open(ranked_json_path, "w") as f:
        json.dump(ranked_data, f, indent=2)

    print(f"Saved ranked data to: {ranked_json_path}")


if __name__ == "__main__":
    main()
