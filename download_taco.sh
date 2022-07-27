#!/bin/bash

git clone https://github.com/tensor-compiler/taco.git
cd taco
git checkout cb00a908865e7122031375622617d9ba65668005
git apply ../fixtaco.patch
cd ..
