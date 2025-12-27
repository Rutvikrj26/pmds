import random
import json
import networkx as nx
import scallopy

class NetworkRoutingGenerator:
    def __init__(self):
        self.ctx = scallopy.ScallopContext()
        self.ctx.import_file("src/scallop_titans/reasoning/rules/routing.scl")
        
        self.TYPES = ["fiber", "wifi", "vpn"]
        self.SECURITIES = ["low", "high"]
        
    def generate_graph(self, num_nodes=20, edge_prob=0.15):
        """Generate a random directed graph with attributes."""
        G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
        nodes = [f"Server_{i}" for i in range(num_nodes)]
        mapping = {i: f"Server_{i}" for i in range(num_nodes)}
        G = nx.relabel_nodes(G, mapping)
        
        edges_data = []
        for u, v in G.edges():
            t = random.choice(self.TYPES)
            s = random.choice(self.SECURITIES)
            edges_data.append((u, v, t, s))
            
        return nodes, edges_data
        
    def solve_reachability(self, edges_data, packet_sec):
        """Use Scallop to compute ground truth reachability."""
        # 1. Reset Context (Create fresh clone or re-import)
        # For simplicity in this script, we just make a new context or clean distinct relations
        ctx = self.ctx.clone()
        
        
        # 2. Add Facts (Types already defined in routing.scl)
        # ctx.add_relation("link", (str, str, str, str))  <-- REMOVED
        # ctx.add_relation("packet", (str,))              <-- REMOVED
        
        ctx.add_facts("link", edges_data)
        ctx.add_facts("packet", [(packet_sec,)])
        
        # 3. Run
        ctx.run()
        
        # 4. Extract Reachable Pairs
        reachable = set()
        for res in ctx.relation("reachable"):
            reachable.add(res) # (source, target)
            
        return reachable

    def generate_sample(self, k_min=4, k_max=8):
        """Generate a single challenging sample."""
        # Retry loop to find non-trivial path
        for _ in range(50):
            nodes, edges = self.generate_graph(num_nodes=30, edge_prob=0.1)
            
            # Pick Random Packet Type
            packet_sec = random.choice(["low", "high"])
            
            # Compute Ground Truth
            reachable_pairs = self.solve_reachability(edges, packet_sec)
            
            if not reachable_pairs: continue
            
            # Pick a pair (A, B)
            # Preference: Pick a pair that is reachable via specific path logic
            # For simplicity, we just pick from reachable set, or non-reachable for negatives
            
            is_positive = random.choice([True, False])
            
            if is_positive:
                if not reachable_pairs: continue
                src, dst = random.choice(list(reachable_pairs))
                if src == dst: continue
                target_bool = True
            else:
                # Pick a pair NOT in reachable
                all_possible = [(u, v) for u in nodes for v in nodes if u != v]
                negatives = [p for p in all_possible if p not in reachable_pairs]
                if not negatives: continue
                src, dst = random.choice(negatives)
                target_bool = False
                
            # Formatting
            # Create text description
            context_text = f"Packet Security Level: {packet_sec.upper()}.\nNetwork Topology:\n"
            facts_list = []
            for u, v, t, s in edges:
                context_text += f"- Link {u} -> {v}: Type={t}, Security={s}\n"
                facts_list.append(f"link({u}, {v}, {t}, {s})")
                
            question = f"Can the packet flow from {src} to {dst}?"
            
            return {
                "context": context_text,
                "question": question,
                "answer": "Yes" if target_bool else "No",
                "scallop_facts": facts_list,
                "packet_sec": packet_sec,
                "src": src,
                "dst": dst
            }
            
        return None

if __name__ == "__main__":
    gen = NetworkRoutingGenerator()
    sample = gen.generate_sample()
    if sample:
        print(json.dumps(sample, indent=2))
    else:
        print("Failed to generate sample")
