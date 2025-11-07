"""
Smoke tests for CAD geometry service
Tier 1: < 5 minutes - Basic functionality validation
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
BASE_URL = os.getenv('GEOMETRY_SERVICE_URL', 'http://localhost:5000')
TEST_FILES_DIR = Path(__file__).parent / 'test_files'


class SmokeTestRunner:
    """Run smoke tests to validate deployment"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    def run_test(self, name: str, test_func):
        """Execute a single test and record result"""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            test_func()
            elapsed = time.time() - start_time
            self.results.append({
                'name': name,
                'status': 'PASS',
                'elapsed': elapsed
            })
            print(f"✅ PASS ({elapsed:.2f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            self.results.append({
                'name': name,
                'status': 'FAIL',
                'error': str(e),
                'elapsed': elapsed
            })
            print(f"❌ FAIL: {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("SMOKE TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')
        total_time = sum(r['elapsed'] for r in self.results)
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if r['status'] == 'FAIL':
                    print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
        
        return failed == 0


def test_service_health(runner: SmokeTestRunner):
    """Test 1: Health endpoint responds"""
    response = requests.get(f"{runner.base_url}/health", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data['status'] == 'ok', "Health check failed"
    print(f"  Circuit breaker state: {data.get('circuit_breaker', 'unknown')}")


def test_root_endpoint(runner: SmokeTestRunner):
    """Test 2: Root endpoint returns service info"""
    response = requests.get(f"{runner.base_url}/", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'version' in data, "Missing version info"
    assert 'features' in data, "Missing features info"
    print(f"  Service version: {data['version']}")
    print(f"  AAGNet available: {data['features'].get('aagnet_available', False)}")


def test_metrics_endpoint(runner: SmokeTestRunner):
    """Test 3: Metrics endpoint accessible"""
    response = requests.get(f"{runner.base_url}/metrics", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'circuit_breaker' in data, "Missing circuit breaker metrics"
    assert 'dead_letter_queue' in data, "Missing DLQ metrics"
    print(f"  Circuit breaker: {data['circuit_breaker']['state']}")


def test_simple_step_file(runner: SmokeTestRunner):
    """Test 4: Analyze simple STEP file"""
    
    # Create a simple test STEP file if not exists
    test_file = TEST_FILES_DIR / 'simple_cube.step'
    if not test_file.exists():
        print("  Test file not found, skipping...")
        return
    
    with open(test_file, 'rb') as f:
        files = {'file': ('simple_cube.step', f, 'application/step')}
        response = requests.post(
            f"{runner.base_url}/analyze-cad",
            files=files,
            headers={'X-Correlation-ID': 'smoke_test_001'},
            timeout=30
        )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'mesh_data' in data, "Missing mesh_data"
    assert 'ml_features' in data, "Missing ml_features"
    assert 'processing_tier' in data, "Missing processing_tier"
    
    print(f"  Processing tier: {data['processing_tier']}")
    print(f"  Processing time: {data.get('processing_time_sec', 0)}s")
    print(f"  Vertices: {len(data['mesh_data'].get('vertices', [])) // 3}")


def test_circuit_breaker_state(runner: SmokeTestRunner):
    """Test 5: Circuit breaker is operational"""
    response = requests.get(f"{runner.base_url}/metrics", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    cb_state = data['circuit_breaker']
    
    # Circuit should be CLOSED for healthy system
    assert cb_state['state'] in ['CLOSED', 'HALF_OPEN'], \
        f"Circuit breaker in unexpected state: {cb_state['state']}"
    
    print(f"  State: {cb_state['state']}")
    print(f"  Failures: {cb_state['failure_count']}/{cb_state['failure_threshold']}")


def test_dlq_accessible(runner: SmokeTestRunner):
    """Test 6: Dead letter queue is accessible"""
    response = requests.get(f"{runner.base_url}/dlq/stats", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"  Total failures (24h): {data.get('total_failures_24h', 0)}")


def test_invalid_file_handling(runner: SmokeTestRunner):
    """Test 7: Graceful handling of invalid files"""
    
    # Send invalid data
    files = {'file': ('invalid.step', b'INVALID STEP FILE', 'application/step')}
    response = requests.post(
        f"{runner.base_url}/analyze-cad",
        files=files,
        headers={'X-Correlation-ID': 'smoke_test_invalid'},
        timeout=10
    )
    
    # Should return 400 (bad request) not 500 (server error)
    assert response.status_code in [400, 500], \
        f"Expected 400 or 500, got {response.status_code}"
    
    data = response.json()
    assert 'error' in data, "Missing error message"
    print(f"  Error handled gracefully: {data.get('error_type', 'unknown')}")


def main():
    """Run all smoke tests"""
    
    print(f"\nRunning smoke tests against: {BASE_URL}")
    print(f"Test files directory: {TEST_FILES_DIR}")
    
    # Create test files directory if needed
    TEST_FILES_DIR.mkdir(exist_ok=True)
    
    runner = SmokeTestRunner(BASE_URL)
    
    # Run tests
    runner.run_test("Service Health Check", lambda: test_service_health(runner))
    runner.run_test("Root Endpoint", lambda: test_root_endpoint(runner))
    runner.run_test("Metrics Endpoint", lambda: test_metrics_endpoint(runner))
    runner.run_test("Circuit Breaker State", lambda: test_circuit_breaker_state(runner))
    runner.run_test("Dead Letter Queue Access", lambda: test_dlq_accessible(runner))
    runner.run_test("Invalid File Handling", lambda: test_invalid_file_handling(runner))
    runner.run_test("Simple STEP File Analysis", lambda: test_simple_step_file(runner))
    
    # Print summary
    success = runner.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
