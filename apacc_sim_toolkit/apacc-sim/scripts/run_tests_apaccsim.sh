#!/bin/bash
#
# run_tests_apaccsim.sh
#
# Comprehensive test runner for APACC-Sim toolkit
# Executes unit tests, integration tests, and optional profiling/linting
#
# Author: George Frangou
# Institution: Cranfield University

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
TEST_TYPE="all"
COVERAGE=true
LINTING=true
PROFILING=false
VERBOSE=false
PARALLEL=true
MARKERS=""
OUTPUT_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section headers
print_header() {
    echo ""
    print_color "$BLUE" "=============================================="
    print_color "$BLUE" "$1"
    print_color "$BLUE" "=============================================="
    echo ""
}

# Help function
show_help() {
    cat << EOF
Usage: ${0##*/} [OPTIONS]

Run comprehensive tests for APACC-Sim toolkit

OPTIONS:
    -t, --type TYPE       Test type: unit, integration, smoke, all (default: all)
    -c, --no-coverage     Disable coverage reporting
    -l, --no-lint         Disable code linting
    -p, --profile         Enable profiling (memory and time)
    -v, --verbose         Verbose test output
    -s, --sequential      Run tests sequentially (default: parallel)
    -m, --markers MARKERS Pytest markers to run specific tests
    -o, --output DIR      Output directory for results (default: ./test_results)
    -h, --help           Show this help message

EXAMPLES:
    ${0##*/}                    # Run all tests with coverage and linting
    ${0##*/} -t unit           # Run only unit tests
    ${0##*/} -t integration -v # Run integration tests with verbose output
    ${0##*/} -p -m slow        # Profile tests marked as slow
    ${0##*/} --no-coverage     # Run tests without coverage report

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -l|--no-lint)
            LINTING=false
            shift
            ;;
        -p|--profile)
            PROFILING=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--sequential)
            PARALLEL=false
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_color "$RED" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
RESULTS_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/test_run.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Print test configuration
print_header "APACC-Sim Test Suite"
echo "Configuration:"
echo "  Test Type: $TEST_TYPE"
echo "  Coverage: $COVERAGE"
echo "  Linting: $LINTING"
echo "  Profiling: $PROFILING"
echo "  Parallel: $PARALLEL"
echo "  Output: $RESULTS_DIR"
echo "  Timestamp: $TIMESTAMP"

# Check Python environment
print_header "Environment Check"
python --version
echo "Python Path: $(which python)"

# Check required packages
print_color "$YELLOW" "Checking required test packages..."
required_packages=("pytest" "pytest-cov" "pytest-xdist" "pylint" "mypy")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python -c "import ${package//-/_}" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    print_color "$RED" "Missing packages: ${missing_packages[*]}"
    print_color "$YELLOW" "Installing missing packages..."
    pip install "${missing_packages[@]}"
fi

# Prepare pytest arguments
PYTEST_ARGS=()

# Verbose output
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v")
else
    PYTEST_ARGS+=("-q")
fi

# Parallel execution
if [ "$PARALLEL" = true ]; then
    # Use number of CPU cores
    NUM_WORKERS=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")
    PYTEST_ARGS+=("-n" "$NUM_WORKERS")
fi

# Coverage options
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=apacc_sim" "--cov-report=html:$RESULTS_DIR/coverage_html" "--cov-report=term" "--cov-report=xml:$RESULTS_DIR/coverage.xml")
fi

# Markers
if [ -n "$MARKERS" ]; then
    PYTEST_ARGS+=("-m" "$MARKERS")
fi

# Output options
PYTEST_ARGS+=("--junitxml=$RESULTS_DIR/junit.xml")
PYTEST_ARGS+=("--html=$RESULTS_DIR/report.html" "--self-contained-html")

# Profiling options
if [ "$PROFILING" = true ]; then
    PYTEST_ARGS+=("--profile" "--profile-svg")
    
    # Memory profiling requires additional setup
    if python -c "import memory_profiler" 2>/dev/null; then
        export PYTEST_MEMPROF=1
    else
        print_color "$YELLOW" "memory_profiler not installed, skipping memory profiling"
    fi
fi

# Run code linting (before tests)
if [ "$LINTING" = true ]; then
    print_header "Code Quality Checks"
    
    # Pylint
    print_color "$YELLOW" "Running pylint..."
    pylint_output="$RESULTS_DIR/pylint_report.txt"
    if pylint --rcfile="$PROJECT_ROOT/.pylintrc" "$PROJECT_ROOT"/*.py "$PROJECT_ROOT"/scripts/*.py > "$pylint_output" 2>&1; then
        print_color "$GREEN" "✓ Pylint passed"
    else
        print_color "$YELLOW" "⚠ Pylint warnings (see $pylint_output)"
    fi
    
    # Type checking with mypy
    print_color "$YELLOW" "Running mypy type checker..."
    mypy_output="$RESULTS_DIR/mypy_report.txt"
    if mypy "$PROJECT_ROOT" --ignore-missing-imports > "$mypy_output" 2>&1; then
        print_color "$GREEN" "✓ Type checking passed"
    else
        print_color "$YELLOW" "⚠ Type checking warnings (see $mypy_output)"
    fi
    
    # Security check with bandit
    if python -c "import bandit" 2>/dev/null; then
        print_color "$YELLOW" "Running security checks..."
        bandit_output="$RESULTS_DIR/bandit_report.txt"
        if bandit -r "$PROJECT_ROOT" -f txt -o "$bandit_output" 2>/dev/null; then
            print_color "$GREEN" "✓ Security checks passed"
        else
            print_color "$YELLOW" "⚠ Security warnings (see $bandit_output)"
        fi
    fi
fi

# Run tests based on type
case $TEST_TYPE in
    unit)
        print_header "Running Unit Tests"
        cd "$PROJECT_ROOT"
        pytest tests/unit "${PYTEST_ARGS[@]}"
        ;;
    
    integration)
        print_header "Running Integration Tests"
        cd "$PROJECT_ROOT"
        pytest tests/integration "${PYTEST_ARGS[@]}"
        ;;
    
    smoke)
        print_header "Running Smoke Tests"
        cd "$PROJECT_ROOT"
        # Quick smoke tests for basic functionality
        pytest tests -k "smoke or quick" "${PYTEST_ARGS[@]}" --maxfail=1
        ;;
    
    all)
        print_header "Running All Tests"
        cd "$PROJECT_ROOT"
        
        # Run tests in order: unit -> integration -> system
        print_color "$YELLOW" "Phase 1: Unit Tests"
        pytest tests/unit "${PYTEST_ARGS[@]}" || true
        
        print_color "$YELLOW" "Phase 2: Integration Tests"
        pytest tests/integration "${PYTEST_ARGS[@]}" || true
        
        print_color "$YELLOW" "Phase 3: System Tests"
        pytest tests/system "${PYTEST_ARGS[@]}" || true
        ;;
    
    *)
        print_color "$RED" "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

# Generate test summary
print_header "Test Summary"

# Parse test results
if [ -f "$RESULTS_DIR/junit.xml" ]; then
    # Extract test statistics from JUnit XML
    python - << EOF
import xml.etree.ElementTree as ET
tree = ET.parse('$RESULTS_DIR/junit.xml')
root = tree.getroot()
testsuite = root.find('.//testsuite')
if testsuite is not None:
    tests = testsuite.get('tests', '0')
    failures = testsuite.get('failures', '0')
    errors = testsuite.get('errors', '0')
    skipped = testsuite.get('skipped', '0')
    time = testsuite.get('time', '0')
    
    print(f"Total Tests: {tests}")
    print(f"Passed: {int(tests) - int(failures) - int(errors) - int(skipped)}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Duration: {float(time):.2f}s")
EOF
fi

# Coverage summary
if [ "$COVERAGE" = true ] && [ -f "$RESULTS_DIR/coverage.xml" ]; then
    echo ""
    print_color "$YELLOW" "Coverage Summary:"
    # Extract coverage percentage
    coverage_percent=$(python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('$RESULTS_DIR/coverage.xml')
root = tree.getroot()
line_rate = float(root.get('line-rate', 0))
print(f'{line_rate * 100:.1f}%')
" 2>/dev/null || echo "N/A")
    echo "  Line Coverage: $coverage_percent"
    echo "  Detailed report: $RESULTS_DIR/coverage_html/index.html"
fi

# Performance profiling summary
if [ "$PROFILING" = true ]; then
    echo ""
    print_color "$YELLOW" "Profiling Results:"
    if [ -f "$RESULTS_DIR/prof/combined.svg" ]; then
        echo "  Performance profile: $RESULTS_DIR/prof/combined.svg"
    fi
    if [ -n "$PYTEST_MEMPROF" ]; then
        echo "  Memory profile: $RESULTS_DIR/prof/memory_report.txt"
    fi
fi

# Create a simple HTML dashboard
print_header "Generating Test Dashboard"

cat > "$RESULTS_DIR/dashboard.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>APACC-Sim Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2E86AB; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { color: #06D6A0; }
        .fail { color: #E63946; }
        .warn { color: #F77F00; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        a { color: #2E86AB; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>APACC-Sim Test Results Dashboard</h1>
EOF

# Add timestamp and configuration
echo "<div class='section'>" >> "$RESULTS_DIR/dashboard.html"
echo "<h2>Test Run Information</h2>" >> "$RESULTS_DIR/dashboard.html"
echo "<p><strong>Timestamp:</strong> $TIMESTAMP</p>" >> "$RESULTS_DIR/dashboard.html"
echo "<p><strong>Test Type:</strong> $TEST_TYPE</p>" >> "$RESULTS_DIR/dashboard.html"
echo "<p><strong>Platform:</strong> $(uname -s) $(uname -m)</p>" >> "$RESULTS_DIR/dashboard.html"
echo "<p><strong>Python:</strong> $(python --version 2>&1)</p>" >> "$RESULTS_DIR/dashboard.html"
echo "</div>" >> "$RESULTS_DIR/dashboard.html"

# Add available reports
echo "<div class='section'>" >> "$RESULTS_DIR/dashboard.html"
echo "<h2>Available Reports</h2>" >> "$RESULTS_DIR/dashboard.html"
echo "<ul>" >> "$RESULTS_DIR/dashboard.html"

if [ -f "$RESULTS_DIR/report.html" ]; then
    echo "<li><a href='report.html'>Pytest HTML Report</a></li>" >> "$RESULTS_DIR/dashboard.html"
fi

if [ -f "$RESULTS_DIR/coverage_html/index.html" ]; then
    echo "<li><a href='coverage_html/index.html'>Coverage Report</a></li>" >> "$RESULTS_DIR/dashboard.html"
fi

if [ -f "$RESULTS_DIR/junit.xml" ]; then
    echo "<li><a href='junit.xml'>JUnit XML Results</a></li>" >> "$RESULTS_DIR/dashboard.html"
fi

if [ -f "$RESULTS_DIR/pylint_report.txt" ]; then
    echo "<li><a href='pylint_report.txt'>Pylint Report</a></li>" >> "$RESULTS_DIR/dashboard.html"
fi

if [ -f "$RESULTS_DIR/mypy_report.txt" ]; then
    echo "<li><a href='mypy_report.txt'>Type Checking Report</a></li>" >> "$RESULTS_DIR/dashboard.html"
fi

echo "</ul></div>" >> "$RESULTS_DIR/dashboard.html"
echo "</body></html>" >> "$RESULTS_DIR/dashboard.html"

print_color "$GREEN" "Dashboard generated: $RESULTS_DIR/dashboard.html"

# CI/CD Integration - Exit codes
print_header "CI/CD Integration"

# Check if any tests failed
if [ -f "$RESULTS_DIR/junit.xml" ]; then
    failures=$(python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('$RESULTS_DIR/junit.xml')
root = tree.getroot()
testsuite = root.find('.//testsuite')
if testsuite is not None:
    failures = int(testsuite.get('failures', '0'))
    errors = int(testsuite.get('errors', '0'))
    print(failures + errors)
else:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ "$failures" -gt 0 ]; then
        print_color "$RED" "Tests FAILED (failures: $failures)"
        
        # In CI mode, exit with error
        if [ -n "$CI" ]; then
            exit 1
        fi
    else
        print_color "$GREEN" "All tests PASSED! ✓"
    fi
fi

# Generate artifacts for CI
if [ -n "$CI" ]; then
    print_color "$YELLOW" "Generating CI artifacts..."
    
    # Create artifacts directory
    mkdir -p "$PROJECT_ROOT/artifacts"
    
    # Copy key files
    cp -r "$RESULTS_DIR"/* "$PROJECT_ROOT/artifacts/" 2>/dev/null || true
    
    # Generate summary for CI
    cat > "$PROJECT_ROOT/artifacts/summary.txt" << EOF
APACC-Sim Test Summary
=====================
Timestamp: $TIMESTAMP
Test Type: $TEST_TYPE
Status: $([ "$failures" -gt 0 ] && echo "FAILED" || echo "PASSED")
Coverage: $coverage_percent

See artifacts/dashboard.html for detailed results.
EOF
fi

print_header "Test Run Complete"
print_color "$BLUE" "Results saved to: $RESULTS_DIR"
print_color "$BLUE" "View dashboard: file://$RESULTS_DIR/dashboard.html"

# Optional: Open dashboard in browser
if command -v xdg-open &> /dev/null && [ -z "$CI" ]; then
    read -p "Open dashboard in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open "$RESULTS_DIR/dashboard.html"
    fi
elif command -v open &> /dev/null && [ -z "$CI" ]; then
    read -p "Open dashboard in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$RESULTS_DIR/dashboard.html"
    fi
fi

exit 0