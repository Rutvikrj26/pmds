"""
Multi-Domain Orchestrator: Generates diverse samples across all domains.

This module provides the main entry point for generating 200K+ samples
with proper distribution across 66 domains using parallel processing.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import Tuple

from scallop_titans.data.generators.domain_registry import (
    get_domain_generator,
    get_all_domains,
    get_all_categories,
    get_domains_by_category,
    print_registry_stats,
)

# Ensure all domains are registered
import scallop_titans.data.generators


@dataclass
class GenerationConfig:
    """Configuration for multi-domain generation."""
    total_samples: int = 200_000
    samples_per_domain: int | None = None  # If None, distribute evenly
    output_dir: Path = Path("data/multi_domain")
    negative_ratio: float = 0.1  # 10% negatives
    distractor_ratio: float = 0.2 # 20% distractors
    seed: int = 42
    workers: int | None = None  # None = use all CPUs
    chunk_size: int = 1000  # Samples per chunk for parallel processing
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.workers is None:
            self.workers = cpu_count()
        
    def get_samples_per_domain(self, num_domains: int) -> int:
        """Calculate samples per domain."""
        if self.samples_per_domain:
            return self.samples_per_domain
        # We don't need to reserve for negatives explicitly anymore since
        # negatives are generated probabilistically per domain call
        return self.total_samples // num_domains


def _generate_domain_samples(args: Tuple[str, int, int, float, float]) -> Tuple[str, list[dict]]:
    """Worker function for parallel generation. Must be at module level for pickle."""
    domain_name, count, seed, neg_ratio, dist_ratio = args
    random.seed(seed)
    
    # Import here to avoid pickle issues
    from scallop_titans.data.generators.domain_registry import get_domain_generator
    
    generator_cls = get_domain_generator(domain_name)
    generator = generator_cls()
    samples = generator.generate_batch(
        count, 
        negative_ratio=neg_ratio, 
        distractor_ratio=dist_ratio
    )
    
    return domain_name, samples


def _generate_chunk(args: Tuple[str, int, int, int]) -> list[dict]:
    """Generate a chunk of samples for a domain."""
    domain_name, chunk_size, chunk_id, seed = args
    random.seed(seed + chunk_id)
    
    from scallop_titans.data.generators.domain_registry import get_domain_generator
    
    generator_cls = get_domain_generator(domain_name)
    generator = generator_cls()
    
    samples = []
    attempts = 0
    max_attempts = chunk_size * 10
    
    while len(samples) < chunk_size and attempts < max_attempts:
        sample = generator.generate_sample()
        if sample:
            samples.append(sample.to_chat_format(generator.config.system_prompt))
        attempts += 1
        
    return samples


class MultiDomainGenerator:
    """
    Orchestrates generation across all registered domains.
    
    Supports parallel generation using multiprocessing for maximum throughput.
    """
    
    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        random.seed(self.config.seed)
        
    def generate_all_parallel(self) -> dict[str, list[dict]]:
        """Generate samples from all domains in parallel."""
        domains = get_all_domains()
        
        if not domains:
            raise RuntimeError("No domains registered! Import domain modules first.")
            
        print(f"Generating from {len(domains)} domains using {self.config.workers} workers...")
        print_registry_stats()
        
        samples_per_domain = self.config.get_samples_per_domain(len(domains))
        print(f"Samples per domain: {samples_per_domain}")
        print(f"Total expected: {samples_per_domain * len(domains):,}")
        
        # Prepare tasks: (domain_name, count, seed, neg_ratio, dist_ratio)
        tasks = [
            (
                domain, 
                samples_per_domain, 
                self.config.seed + i,
                self.config.negative_ratio,
                self.config.distractor_ratio
            ) 
            for i, domain in enumerate(domains)
        ]
        
        all_samples: dict[str, list[dict]] = {}
        start_time = time.time()
        
        with Pool(processes=self.config.workers) as pool:
            results = pool.imap_unordered(_generate_domain_samples, tasks)
            
            for domain_name, samples in results:
                all_samples[domain_name] = samples
                elapsed = time.time() - start_time
                total_so_far = sum(len(s) for s in all_samples.values())
                rate = total_so_far / elapsed if elapsed > 0 else 0
                print(f"[{domain_name}] {len(samples):,} samples | "
                      f"Total: {total_so_far:,} | Rate: {rate:.0f}/s")
                
        elapsed = time.time() - start_time
        total = sum(len(s) for s in all_samples.values())
        print(f"\nGeneration complete: {total:,} samples in {elapsed:.1f}s "
              f"({total/elapsed:.0f} samples/sec)")
              
        return all_samples
    
    def generate_all(self) -> dict[str, list[dict]]:
        """Generate samples - uses parallel by default."""
        if self.config.workers > 1:
            return self.generate_all_parallel()
        else:
            return self._generate_all_sequential()
            
    def _generate_all_sequential(self) -> dict[str, list[dict]]:
        """Generate samples sequentially (for debugging)."""
        domains = get_all_domains()
        
        if not domains:
            raise RuntimeError("No domains registered! Import domain modules first.")
            
        print(f"Generating from {len(domains)} domains (sequential mode)...")
        print_registry_stats()
        
        samples_per_domain = self.config.get_samples_per_domain(len(domains))
        print(f"Samples per domain: {samples_per_domain}")
        
        all_samples: dict[str, list[dict]] = {}
        
        for domain_name in domains:
            print(f"\n[{domain_name}] Generating {samples_per_domain} samples...")
            
            generator_cls = get_domain_generator(domain_name)
            generator = generator_cls()
            
            samples = generator.generate_batch(
                samples_per_domain,
                negative_ratio=self.config.negative_ratio,
                distractor_ratio=self.config.distractor_ratio
            )
            all_samples[domain_name] = samples
            
            print(f"[{domain_name}] Generated {len(samples)} samples")
            
        return all_samples
    
    def generate_and_save(self, combined: bool = True) -> Path:
        """Generate all samples and save to files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        all_samples = self.generate_all()
        
        # Save per-domain files in parallel
        print("\nSaving per-domain files...")
        for domain_name, samples in all_samples.items():
            domain_path = self.config.output_dir / f"{domain_name}.jsonl"
            with open(domain_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            print(f"  {domain_name}: {len(samples):,} samples")
                    
        # Save combined shuffled file
        if combined:
            combined_path = self.config.output_dir / "combined_multi_domain.jsonl"
            all_flat = []
            for samples in all_samples.values():
                all_flat.extend(samples)
                
            print(f"\nShuffling {len(all_flat):,} samples...")
            random.shuffle(all_flat)
            
            print(f"Writing combined file...")
            with open(combined_path, "w") as f:
                for s in all_flat:
                    f.write(json.dumps(s) + "\n")
                    
            print(f"✓ Saved {len(all_flat):,} combined samples to {combined_path}")
            return combined_path
            
        return self.config.output_dir
    
    def generate_preview(self, samples_per_domain: int = 3) -> list[dict]:
        """Generate a small preview from each domain for review."""
        domains = get_all_domains()
        preview = []
        
        for domain_name in domains:
            generator_cls = get_domain_generator(domain_name)
            generator = generator_cls()
            
            for _ in range(samples_per_domain):
                sample = generator.generate_sample()
                if sample:
                    preview.append({
                        "domain": domain_name,
                        "sample": sample.to_chat_format()
                    })
                    
        return preview
    
    def print_preview(self, samples_per_domain: int = 2):
        """Print a formatted preview of samples from each domain."""
        domains = get_all_domains()
        
        print("=" * 80)
        print("MULTI-DOMAIN GENERATOR PREVIEW")
        print("=" * 80)
        
        for domain_name in domains:
            print(f"\n{'─' * 40}")
            print(f"Domain: {domain_name}")
            print(f"{'─' * 40}")
            
            generator_cls = get_domain_generator(domain_name)
            generator = generator_cls()
            
            for i in range(samples_per_domain):
                sample = generator.generate_sample()
                if sample:
                    print(f"\n[Sample {i+1}]")
                    print(f"Story: {sample.story}")
                    print(f"Question: {sample.question}")
                    print(f"Scallop: {sample.scallop_cmd}")
                    print(f"Answer: {sample.answer}")


def main():
    """CLI entry point for multi-domain generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Domain Data Generator")
    parser.add_argument("--total", type=int, default=200_000, help="Total samples")
    parser.add_argument("--output", type=str, default="data/multi_domain", help="Output dir")
    parser.add_argument("--preview", action="store_true", help="Just show preview")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: all CPUs)")
    parser.add_argument("--sequential", action="store_true", help="Force sequential mode")
    parser.add_argument("--negative-ratio", type=float, default=0.1, help="Ratio of unanswerable samples")
    parser.add_argument("--distractor-ratio", type=float, default=0.2, help="Ratio of samples with noise")
    args = parser.parse_args()
    
    workers = 1 if args.sequential else args.workers
    
    config = GenerationConfig(
        total_samples=args.total,
        output_dir=Path(args.output),
        seed=args.seed,
        workers=workers,
        negative_ratio=args.negative_ratio,
        distractor_ratio=args.distractor_ratio,
    )
    
    print(f"Configuration:")
    print(f"  Total samples: {config.total_samples:,}")
    print(f"  Workers: {config.workers}")
    print(f"  Output: {config.output_dir}")
    print(f"  Negative Ratio: {config.negative_ratio:.2f}")
    print(f"  Distractor Ratio: {config.distractor_ratio:.2f}")
    print()
    
    generator = MultiDomainGenerator(config)
    
    if args.preview:
        generator.print_preview()
    else:
        generator.generate_and_save()


if __name__ == "__main__":
    main()
