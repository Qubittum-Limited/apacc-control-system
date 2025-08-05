#!/bin/bash
# APACC Test Suite Runner
# =======================
# Runs all pytest-based tests with coverage reporting and optional profiling
#
# Usage:
#   ./run_tests.sh                    # Run all tests with coverage
#   ./run_tests.sh --profile          # Run with profiling enabled
#   ./run_tests.sh --quick            # Run only unit tests (skip integration)
#   ./run_tests.sh --verbose          # Run with verbose output
#
# Author: George Frangou
# Institution: Cranfield University
# DOI: https://doi.org/10.5281/zenodo.8475

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default values
PROFILE=false
QUICK=false
VERBOSE=""
PYTEST_ARGS=""
COVERAGE_THRESHOLD=80

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --verbose|-v)
            VERBOSE="-vv"
            shift
            ;;
        --help|-h)
            echo "APACC Test Suite Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --profile    Enable performance profiling"
            echo "  --quick      Run only unit tests (skip integration)"
            echo "  --verbose    Enable verbose output"
            echo "  --help       Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}APACC Validation Test Suite${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Please install pytest: pip install pytest pytest-cov"
    exit 1
fi

# Create test results directory
RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Created test results directory: ${RESULTS_DIR}${NC}"

# Configure pytest arguments
PYTEST_ARGS="--maxfail=2 --disable-warnings --tb=short"
PYTEST_ARGS="$PYTEST_ARGS --cov=. --cov-report=html:${RESULTS_DIR}/htmlcov"
PYTEST_ARGS="$PYTEST_ARGS --cov-report=term-missing"
PYTEST_ARGS="$PYTEST_ARGS --cov-report=xml:${RESULTS_DIR}/coverage.xml"
PYTEST_ARGS="$PYTEST_ARGS --junit-xml=${RESULTS_DIR}/test_results.xml"
PYTEST_ARGS="$PYTEST_ARGS $VERBOSE"

# Add test markers based on options
if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}Running in quick mode (unit tests only)${NC}"
    PYTEST_ARGS="$PYTEST_ARGS -m 'not integration'"
fi

# Environment validation
echo ""
echo -e "${BOLD}Running environment validation...${NC}"
if python scripts/validate_environment.py; then
    echo -e "${GREEN}Environment validation passed${NC}"
else
    echo -e "${YELLOW}Warning: Environment validation reported issues${NC}"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run unit tests
echo ""
echo -e "${BOLD}Running unit tests...${NC}"
echo "Command: pytest tests/unit/ $PYTEST_ARGS"

if [ -d "tests/unit" ]; then
    pytest tests/unit/ $PYTEST_ARGS 2>&1 | tee "${RESULTS_DIR}/unit_test_output.log"
    UNIT_EXIT_CODE=${PIPESTATUS[0]}
else
    echo -e "${YELLOW}Warning: tests/unit/ directory not found${NC}"
    UNIT_EXIT_CODE=0
fi

# Run integration tests (unless in quick mode)
if [ "$QUICK" = false ]; then
    echo ""
    echo -e "${BOLD}Running integration tests...${NC}"
    echo "Command: pytest tests/integration/ $PYTEST_ARGS"
    
    if [ -d "tests/integration" ]; then
        pytest tests/integration/ $PYTEST_ARGS 2>&1 | tee "${RESULTS_DIR}/integration_test_output.log"
        INTEGRATION_EXIT_CODE=${PIPESTATUS[0]}
    else
        echo -e "${YELLOW}Warning: tests/integration/ directory not found${NC}"
        INTEGRATION_EXIT_CODE=0
    fi
else
    INTEGRATION_EXIT_CODE=0
fi

# Run all tests if no specific test directories exist
if [ ! -d "tests/unit" ] && [ ! -d "tests/integration" ]; then
    echo ""
    echo -e "${BOLD}Running all tests...${NC}"
    echo "Command: pytest tests/ $PYTEST_ARGS"
    
    if [ -d "tests" ]; then
        pytest tests/ $PYTEST_ARGS 2>&1 | tee "${RESULTS_DIR}/all_tests_output.log"
        ALL_EXIT_CODE=${PIPESTATUS[0]}
    else
        echo -e "${RED}Error: No tests directory found${NC}"
        echo "Please create tests/ directory with test files"
        exit 1
    fi
else
    ALL_EXIT_CODE=0
fi

# Performance profiling (if requested)
if [ "$PROFILE" = true ]; then
    echo ""
    echo -e "${BOLD}Running performance profiling...${NC}"
    
    # Check if profiling tools are available
    if python -c "import line_profiler, memory_profiler" 2>/dev/null; then
        # Profile main runner script
        echo "Profiling runner.py..."
        python -m memory_profiler runner.py --env monte_carlo --episodes 10 > "${RESULTS_DIR}/memory_profile.txt" 2>&1 || true
        
        # Line profiling for critical functions
        if [ -f "scripts/profile_critical_paths.py" ]; then
            python scripts/profile_critical_paths.py > "${RESULTS_DIR}/line_profile.txt" 2>&1 || true
        fi
        
        echo -e "${GREEN}Profiling results saved to ${RESULTS_DIR}/${NC}"
    else
        echo -e "${YELLOW}Warning: Profiling tools not installed${NC}"
        echo "Install with: pip install line-profiler memory-profiler"
    fi
fi

# Generate coverage report
echo ""
echo -e "${BOLD}Generating coverage report...${NC}"

# Check coverage threshold
COVERAGE_PERCENT=$(python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('${RESULTS_DIR}/coverage.xml')
    root = tree.getroot()
    coverage = float(root.attrib.get('line-rate', 0)) * 100
    print(f'{coverage:.1f}')
except:
    print('0.0')
")

echo -e "Overall test coverage: ${BOLD}${COVERAGE_PERCENT}%${NC}"

if (( $(echo "$COVERAGE_PERCENT < $COVERAGE_THRESHOLD" | bc -l) )); then
    echo -e "${YELLOW}Warning: Coverage ${COVERAGE_PERCENT}% is below threshold of ${COVERAGE_THRESHOLD}%${NC}"
else
    echo -e "${GREEN}Coverage meets threshold of ${COVERAGE_THRESHOLD}%${NC}"
fi

# Summary
echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}Test Summary${NC}"
echo -e "${BOLD}========================================${NC}"

# Determine overall status
OVERALL_EXIT_CODE=$((UNIT_EXIT_CODE + INTEGRATION_EXIT_CODE + ALL_EXIT_CODE))

if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

echo ""
echo "Test results saved to: ${RESULTS_DIR}/"
echo "  - Test output logs: *_output.log"
echo "  - Coverage report: htmlcov/index.html"
echo "  - JUnit XML: test_results.xml"
echo "  - Coverage XML: coverage.xml"

if [ "$PROFILE" = true ]; then
    echo "  - Memory profile: memory_profile.txt"
    echo "  - Line profile: line_profile.txt"
fi

echo ""
echo -e "${BOLD}View coverage report:${NC}"
echo "  cd ${RESULTS_DIR} && python -m http.server 8000"
echo "  Then open: http://localhost:8000/htmlcov/"

# Create test summary file
cat > "${RESULTS_DIR}/test_summary.txt" << EOF
APACC Test Suite Summary
========================
Date: $(date)
Coverage: ${COVERAGE_PERCENT}%
Unit Tests: $([ $UNIT_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED")
Integration Tests: $([ $INTEGRATION_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED")
Profile Mode: $([ "$PROFILE" = true ] && echo "ENABLED" || echo "DISABLED")

Configuration:
- Python: $(python --version 2>&1)
- pytest: $(pytest --version 2>&1 | head -1)
- Platform: $(uname -a)

Results Directory: ${RESULTS_DIR}
EOF

echo ""
echo "Test summary written to: ${RESULTS_DIR}/test_summary.txt"

# Exit with appropriate code
exit $OVERALL_EXIT_CODE