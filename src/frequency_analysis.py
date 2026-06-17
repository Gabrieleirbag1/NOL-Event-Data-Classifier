from collections import defaultdict

def find_frequent_words_across_clusters(clusters):
    """
    Si un mot apparaît dans BEAUCOUP de clusters différents,
    c'est probablement un paramètre, pas un concept métier.
    """
    word_to_clusters = defaultdict(set)

    for cluster_name, events in clusters.items():
        for event in events:
            for word in event.lower().split():
                word_to_clusters[word].add(cluster_name)

    # Mots présents dans plus de N clusters distincts = suspects
    suspects = {
        word: len(cluster_set)
        for word, cluster_set in word_to_clusters.items()
        if len(cluster_set) >= 3   # seuil à ajuster
    }

    return sorted(suspects.items(), key=lambda x: -x[1])

def run_frequent_params_analysis(clusters):
    suspects = find_frequent_words_across_clusters(clusters)
    for word, n_clusters in suspects[:20]:
        print(f"{word}: apparaît dans {n_clusters} clusters")