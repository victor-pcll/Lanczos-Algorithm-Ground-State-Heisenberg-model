import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .plot_params import * 

def plot_netket_graph(graph, layout='circular', edge_styles=None, 
                      node_size=600, node_color='#e0e1dd', font_color='black',
                      title=None, save_path=None, figsize=(8, 8), dpi=120):
    """
    Trace un graphe NetKet de manière propre et paramétrable.
    
    Arguments:
    - graph : L'objet nk.graph.Graph de NetKet.
    - layout (str) : Disposition ('circular', 'spring', 'kamada_kawai', 'square_pbc').
    - edge_styles (dict) : Dictionnaire associant l'ID de couleur NetKet à son style visuel.
    - node_size, node_color, font_color : Esthétique des noeuds.
    - title (str) : Titre optionnel du graphique.
    - save_path (str) : Chemin complet pour sauvegarder l'image.
    - figsize (tuple) : Taille de la figure.
    - dpi (int) : Résolution pour l'affichage et la sauvegarde.
    """
    nx_g = graph.to_networkx()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # ==========================================
    # 1. CAS SPÉCIAL : GRILLE AVEC PBC (SQUARE_PBC)
    # ==========================================
    if layout == 'square_pbc':
        if not hasattr(graph, 'positions'):
            raise ValueError("Le graphe NetKet n'a pas de positions définies, impossible d'utiliser 'square_pbc'.")
            
        # Extraction des coordonnées exactes (gère 1D et 2D)
        pos = {}
        for i, p in enumerate(graph.positions):
            x = p[0]
            y = p[1] if len(p) > 1 else 0.0
            pos[i] = (x, y)
            
        min_x, max_x = min(p[0] for p in pos.values()), max(p[0] for p in pos.values())
        min_y, max_y = min(p[1] for p in pos.values()), max(p[1] for p in pos.values())
        
        # A. Filtrage des arêtes (On sépare le "Bulk" des "Wrap-around" PBC)
        bulk_edges = []
        for u, v, d in nx_g.edges(data=True):
            dist_x = abs(pos[u][0] - pos[v][0])
            dist_y = abs(pos[u][1] - pos[v][1])
            # Si la distance est grande, c'est une arête qui traverse le graphe (PBC)
            if dist_x < (max_x - min_x) * 0.8 and dist_y < (max_y - min_y) * 0.8:
                bulk_edges.append((u, v, d))
                
        # B. Création des "Stubs" (Petits pointillés vers l'extérieur)
        stub_segments = []
        STUB_LENGTH = 0.4
        
        for node, (x, y) in pos.items():
            if x == min_x: stub_segments.append([(x, y), (x - STUB_LENGTH, y)])
            if x == max_x: stub_segments.append([(x, y), (x + STUB_LENGTH, y)])
            if y == min_y: stub_segments.append([(x, y), (x, y - STUB_LENGTH)])
            if y == max_y: stub_segments.append([(x, y), (x, y + STUB_LENGTH)])

        stubs_collection = LineCollection(stub_segments, linewidths=2.0, linestyles="dashed", 
                                          colors="gray", alpha=0.7)
        ax.add_collection(stubs_collection)
        
        # C. Dessin des arêtes internes (Bulk)
        if edge_styles is None:
            nx.draw_networkx_edges(nx_g, pos, edgelist=[(u,v) for u,v,d in bulk_edges], 
                                   width=2.5, edge_color='gray', ax=ax)
        else:
            for edge_color_id, style in edge_styles.items():
                edges_of_type = [(u, v) for u, v, d in bulk_edges if d.get('color') == edge_color_id]
                if edges_of_type:
                    nx.draw_networkx_edges(
                        nx_g, pos, edgelist=edges_of_type,
                        width=style.get('width', 2.0), alpha=style.get('alpha', 0.8),
                        edge_color=style.get('color', 'gray'), style=style.get('style', '-'),
                        label=style.get('label', f'Interaction {edge_color_id}'), ax=ax
                    )
        
        ax.set_aspect('equal')
        margin = STUB_LENGTH * 1.5
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

    # ==========================================
    # 2. CAS CLASSIQUES (Circular, Spring, etc.)
    # ==========================================
    else:
        if layout == 'circular':
            pos = nx.circular_layout(nx_g)
        elif layout == 'spring':
            pos = nx.spring_layout(nx_g, k=0.5, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(nx_g)
        else:
            raise ValueError("Layout non reconnu. Choisissez 'circular', 'spring', 'kamada_kawai' ou 'square_pbc'.")

        if edge_styles is None:
            nx.draw_networkx_edges(nx_g, pos, width=2.0, alpha=0.8, edge_color='gray', ax=ax)
        else:
            for edge_color_id, style in edge_styles.items():
                edges_of_type = [(u, v) for u, v, d in nx_g.edges(data=True) if d.get('color') == edge_color_id]
                if edges_of_type:
                    nx.draw_networkx_edges(
                        nx_g, pos, edgelist=edges_of_type,
                        width=style.get('width', 2.0), alpha=style.get('alpha', 0.8),
                        edge_color=style.get('color', 'gray'), style=style.get('style', '-'),
                        label=style.get('label', f'Interaction {edge_color_id}'), ax=ax
                    )

    # ==========================================
    # 3. DESSIN DES NOEUDS, LABELS ET FORMATAGE FINAL
    # ==========================================
    nx.draw_networkx_nodes(nx_g, pos, node_size=node_size, node_color=node_color, 
                           edgecolors='black', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(nx_g, pos, font_size=11, font_weight='bold', font_color=font_color, ax=ax)

    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        
    if edge_styles:
        plt.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.9, edgecolor="black")
        
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphe sauvegardé avec succès sous : {save_path}")
        
    plt.show()