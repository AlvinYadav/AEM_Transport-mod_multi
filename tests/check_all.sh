#!/usr/bin/env bash

run_ruff() {
    echo "ruff:"
    ruff check src/
    ruff check tests/

    echo -e "\n\n-------------------------------------------\n\n"
}

run_mypy() {
    echo "mypy:"
    mypy --check-untyped-defs src/
    mypy --check-untyped-defs tests/

    echo -e "\n\n-------------------------------------------\n\n"
}

run_flake8() {
    echo "flake8:"
    flake8 src/
    flake8 tests/

    echo -e "\n\n-------------------------------------------\n\n"
}

run_pyright() {
    echo "pyright:"
    pyright src/
    pyright tests/
}

reset

export PYTHONPATH=$PYTHONPATH:"src/"

run_ruff
run_mypy
run_flake8
run_pyright
