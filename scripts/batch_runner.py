"""
Batch Watermark Execution Engine with Throughput Profiling.
Runs watermark embedding across collections of image buffers using thread worker pools.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Callable

class BatchPipelineRunner:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_batch(
        self,
        items: List[dict],
        embed_fn: Callable[[List[int], str], List[int]],
        message: str
    ) -> Dict:
        """
        Processes items in parallel.
        items: list of {'id': str, 'pixels': List[int]}
        """
        start_time = time.perf_counter()
        results = []
        errors = []

        def worker(item):
            try:
                watermarked = embed_fn(item['pixels'], message)
                return {'id': item['id'], 'success': True, 'pixels': watermarked}
            except Exception as e:
                return {'id': item['id'], 'success': False, 'error': str(e)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(worker, item): item['id'] for item in items}
            for future in as_completed(future_to_id):
                res = future.result()
                if res['success']:
                    results.append(res)
                else:
                    errors.append(res)

        elapsed = time.perf_counter() - start_time
        throughput = len(items) / elapsed if elapsed > 0 else 0.0

        return {
            'total_items': len(items),
            'succeeded': len(results),
            'failed': len(errors),
            'elapsed_sec': round(elapsed, 4),
            'throughput_items_per_sec': round(throughput, 2),
            'results': results,
            'errors': errors,
        }
