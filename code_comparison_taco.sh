#!/bin/bash

export DYLD_FALLBACK_LIBRARY_PATH="./taco/build/lib"
./taco/build/bin/taco "C = A(i) * B(i)" -f=A:s:0 -f=B:s:0